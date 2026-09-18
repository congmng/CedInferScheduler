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

| 检查 | 4090 (sm89) | 5090 (sm120) |
|---|---|---|
| 参数（模块树） | 831.3M | 831.3M |
| 与估算器之差 | 5.2% | 5.2% |
| KV/token | 8.438 KB | 8.438 KB |
| logits checksum | -257011226.4 | -257010971.9（相对差 1e-6） |

冒烟档的层结构：ratio-0 层 window 128 / states 0；ratio-4 层 states 32(每 128 token)、
window 8；ratio-128 层 states 1、window 128 —— 与 `coff·ratio` 的定义一致。

## 设计上踩过并修掉的两个坑

1. **compressor 不能对整窗做 dense 投影**（参数 `hidden×window×state`，
   HCA 窗口 128 时每层 134M 参数）。改成"窗口池化 + `hidden→state` 小投影"。
2. **整行无效的 top-k 会产出 NaN**（softmax 全 `-inf`）：用有限大负数掩码，
   并把该块贡献置零。
