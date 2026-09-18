# dsv4_ref — 可移植的 DSV4 风格模型参考实现

纯 PyTorch（无自定义 kernel、无 fp8、无 tilelang、**无 mHC**），用途有三：

1. **设计规格**：`config.py` 定义层模式（`[0,0]+[4,128]×k+[0]`）、MLA latent 维度、
   压缩比、indexer top-k，以及参数量/KV 的解析估算；
   `model.py` 是这些规格的可执行版本（标准 pre-norm 残差，已砍 mHC）。
2. **KV 账目核算**：`verify.py` 逐层数一个 token 实际占多少字节，与闭式
   （全层 `head_dim+rope`、压缩层 `2·coff·head_dim/ratio`）对照。
3. **跨卡证据**：同一份代码在 4090（sm89）与 5090（sm120）上跑同一输入并比较
   logits 指纹。

## 用法

```bash
# 在 vLLM 镜像里跑（宿主 python 没有 torch）
docker run --rm --gpus '"device=1"' --entrypoint python3 -v "$PWD":/work -w /work \
  vllm/vllm-openai:casr029 -m design.dsv4_ref.verify \
  --config small --tokens 128 --device cuda --dtype float32

# 只算规模与 KV（不建模型，适合放不下的档位）
docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \
  vllm/vllm-openai:casr029 -m design.dsv4_ref.verify --config p15b --summary-only
```

## 已记录的实测（2026-09-18）

`--config small --tokens 128`，fp32：

| 检查 | 3090 (sm86) | 4090 (sm89) | 5090 (sm120) |
|---|---|---|---|
| 参数（模块树） | 831,341,312 | 同 | 同 |
| 与估算器之差 | 5.2% | 5.2% | 5.2% |
| KV/token | 8.438 KB | 8.438 KB | 8.438 KB |
| logits checksum | -257011068.9 | -257011226.4 | -257010971.9 |
| 与 4090 相对差 | 6.1e-7 | — | 9.9e-7 |

**Ampere / Ada / Blackwell 三档一致到 1e-6**（差异只是归约顺序）。
bf16 也跑通了（4090 checksum `-257179527.3`）；修掉的那条 fp32 泄漏见
`model.py` 的 `rope()`：角度在 fp32 里算完必须 cast 回激活 dtype，
否则整条注意力路径被上采成 fp32，bf16 权重会在第一个 `o_lora_a` 报 dtype 错。

冒烟档的层结构：ratio-0 层 window 128 / states 0；ratio-4 层 states 32(每 128 token)、
window 8；ratio-128 层 states 1、window 128 —— 与 `coff·ratio` 的定义一致。

## 四档配置（`REF_CONFIGS`，`--summary-only` 直接报出）

结构一律是 MLA latent + single-head indexer + compressor + MoE，词表 64k，
ratio 走官方交替模式。

| 档 | 层/hidden/heads | MoE | 总参 / 激活 | KV/token (bf16 / fp8) | 权重 bf16 / fp8 |
|---|---|---:|---:|---:|---:|
| `small` | 12 / 1024 / 16 | 32, top-2 | 0.877B / 0.176B | 8.44 / 4.22 KB | 1.75 / 0.88 GB |
| `p5b` | 20 / 1792 / 14 | 40, top-2 | 5.131B / 0.712B | 12.50 / 6.25 KB | 10.26 / 5.13 GB |
| `p15b` | 28 / 2560 / 20 | 48, top-3 | 14.866B / 2.145B | 16.56 / 8.28 KB | 29.73 / 14.87 GB |
| `p29b` | 30 / 3072 / 24 | 64, top-3 | 29.610B / 3.302B | 17.58 / 8.79 KB | 59.22 / 29.61 GB |

**估算器 vs 模块树（`small`，逐项对过账）**：估算 877,000,000，模块树
831,341,312 → **估算高 5.2%**。差在三处：
① 估算器给每层都计 `index_n_heads` 头的 indexer（31.5M），而模块里 indexer 是
**单头**且只在压缩层（1.21M）→ 高估 30.2M；
② 估算器没有 `kv_state_proj`（compress state → `head_dim`），模块里有 3.67M → 低估；
③ MoE 估算比模块高 1.57M/层 → 高估 18.9M。

## 设计上踩过并修掉的两个坑

1. **compressor 不能对整窗做 dense 投影**（参数 `hidden×window×state`，
   HCA 窗口 128 时每层 134M 参数）。改成"窗口池化 + `hidden→state` 小投影"。
2. **整行无效的 top-k 会产出 NaN**（softmax 全 `-inf`）：用有限大负数掩码，
   并把该块贡献置零。
