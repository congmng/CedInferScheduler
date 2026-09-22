# P-15B 具体设计（2026-09-22）

> 这是**自研目标模型的实现规格**，不是可行性调研（那份见
> [DSV4_10-30B跨卡设计.md](DSV4_10-30B跨卡设计.md)）。本文所有数字都取自
> `LLMServingSim/design/dsv4_ref/`，复现命令见 §9；与代码冲突时以代码为准。
>
> **一句话**：28 层、hidden 2560、总参 **14.376B / 激活 1.655B** 的 MoE 模型，
> 注意力用 MLA + CSA/HCA 混合压缩，**KV `16.56 KB/token`（bf16）**——比
> Qwen3-8B 的 `147.5 KB/token` 小 8.9×；没有 mHC，权重 **28.75 GB（bf16）**，
> 四类卡（3090/4090/5090/A100）上都能跑。

---

## 1. 硬约束（这些决定了下面的每个数）

| # | 约束 | 后果 |
|---|---|---|
| 1 | 要在 3090(sm86) / 4090(sm89) / 5090(sm120) / A100(sm80) 上跑 | 不能有 mHC（只有 sm90+ 的 deep_gemm kernel），不能依赖 fp8 GEMM（sm80/sm86 没有） |
| 2 | KV 必须比 Qwen3-8B 小一个量级 | 逐层压缩比 CSA(4)/HCA(128) 混合，不是靠减层 |
| 3 | 单卡 24GB 装不下 28.75 GB 权重 | 部署形态是 bf16 + 同机 TP=2（14.38 GB/卡） |
| 4 | 研究目标是调度，不是刷榜 | MoE 用中等规模（48 专家 top-3），不做 hash 层等训练技巧 |

## 2. 规格

| 项 | 值 | 备注 |
|---|---|---|
| 层数 | **28** | `compress_ratios` 28 项 |
| hidden | **2560** | |
| attention heads | **20** | `head_dim=512` → q 维 10240 |
| KV heads | **1** | MLA latent，单头 |
| `head_dim` / `qk_rope_head_dim` | **512 / 64** | 官方 MLA 取值 |
| `q_lora_rank` / `o_lora_rank` / `o_groups` | **640 / 512 / 8** | 分组输出 LoRA |
| indexer `head_dim` / `n_heads` / `topk` | **128 / 20 / 512** | 参考实现里 indexer 是单头（见 §6） |
| MoE | **48 routed，top-3**，`moe_intermediate_size=1280` | 无 shared expert、无 hash 层 |
| RoPE | `theta=10000`，`max_position_embeddings=4096` | 先短上下文，长上下文另议 |
| vocab | **65536** | 官方 129280；换小省 ~0.17B 但 tokenizer 要自己训 |
| norm | RMSNorm，`eps=1e-6`，pre-norm | 无 mHC |
| 总参 / 激活 | **14.376 B / 1.655 B** | 模块树实测（§4） |
| 权重 | **28.75 GB** bf16 / **14.38 GB** fp8 | |
| KV/token | **16.56 KB** bf16 / **8.28 KB** fp8 | 逐层闭式（§5） |

## 3. 层结构

沿用官方交替模式 `[0,0] + [4,128]×k + [0]`，28 层展开为：

```text
idx   0  1 | 2   3 | 4   5 | 6   7 | 8   9 | 10  11 | 12  13 | 14  15
      0  0 | 4 128 | 4 128 | 4 128 | 4 128 | 4   128 | 4   128 | 4   128

idx  16 17 | 18  19 | 20  21 | 22  23 | 24  25 | 26 27
      4 128 | 4   128 | 4   128 | 4   128 | 4   128 | 4  0
```

- **3 层全 MLA**（idx 0/1/27）：整段上下文都存 latent
- **13 层 CSA**（ratio 4）：每 4 个 token 压成一个状态，滑窗 `coff×ratio = 8`
- **12 层 HCA**（ratio 128）：每 128 个 token 压成一个状态，滑窗 128

每层内部（`DecoderLayer`）：

```text
x = x + Attention(RMSNorm(x))      # MLA + 滑窗 + 压缩状态（+ indexer top-k）
x = x + MoE(RMSNorm(x))            # 标准 pre-norm 残差，无 mHC
```

Attention 内部三件事：

1. **MLA latent**：`qkv_down(hidden → q_lora + head_dim + rope)`，q 走 `q_up` 展开到
   20 头；kv latent 宽 `head_dim`，全层每 token 存一次。
2. **滑窗**：对最近 `coff×ratio` 个 token 的 raw latent 做 banded attention
   （ratio 0 层退化为全因果）。
3. **压缩状态**：`compressor` 先**窗口池化**再投影（不能对整窗做 dense 投影，
   否则 HCA 每层 134M 参数），产出 `2·coff·head_dim` 宽的状态；`indexer` 打分取
   top-512；**窗口与选中状态合成一个拼接后的联合 softmax**（这条踩过坑，见 §6）。

## 4. 参数量分解（模块树实测）

单层按层类型（attention 部分随 ratio 变，MoE 部分与 ratio 无关）：

| 组件 | 全 MLA (×3) | CSA ratio 4 (×13) | HCA ratio 128 (×12) |
|---|---:|---:|---:|
| `qkv_down` | 3,112,960 | 3,112,960 | 3,112,960 |
| `q_up` | 6,553,600 | 6,553,600 | 6,553,600 |
| `o_lora`（8 组 + wo_b） | 15,728,640 | 15,728,640 | 15,728,640 |
| `compressor` | 0 | 5,244,928 | 2,622,464 |
| `indexer` | 0 | 212,992 | 147,456 |
| `kv_state_proj` | 0 | 524,288 | 262,144 |
| norms（4 个） | 6,336 | 6,336 | 6,336 |
| **MoE**（gate + 48 专家） | **471,982,080** | **471,982,080** | **471,982,080** |
| **单层合计** | **497,383,616** | **503,365,824** | **500,415,680** |

总量：

```text
28 层                        14,040,894,720
embed + lm_head                 335,544,320
final norm                            2,560
────────────────────────────────────────────
合计                         14,376,441,600  = 14.376 B（bf16 28.75 GB）
```

**MoE 占 91.9%**（13.21B / 14.38B）。激活量按 top-3 算：

```text
每层激活 = attention + norms + gate + 3×专家参数
ratio0     55,015,616 ×3   =   165,046,848
ratio4     60,997,824 ×13  =   792,971,712
ratio128   58,047,680 ×12  =   696,572,160
──────────────────────────────────────────
激活合计                       1,654,590,720 = 1.655 B
```

> ⚠️ 配置估算器（`config.summary()`）报的是 **14.866 B / 2.145 B**，与模块树差
> 0.49 B，来源三处、已逐项查过：① 估算器给每层都算了一个 **shared expert**
> （模块里没有，+0.275 B）；② 估算器按 `index_n_heads=20` 算 indexer，而实现是
> **单头**（+0.225 B）；③ 估算器没有 `kv_state_proj`（−0.010 B）。
> **显存与训练预算一律以模块树为准**，估算器只用来快速比规模。

## 5. KV 记账

逐层闭式（值数 × 字节数，bf16 = 2 B）：

| 层类型 | 每状态宽 | 每多少 token 一个状态 | KV/token |
|---|---:|---:|---:|
| 全 MLA | `head_dim + rope` = 576 值 | 1 | 1,152 B |
| CSA（ratio 4） | `2·coff·head_dim` = 2048 值 | 4 | 1,024 B |
| HCA（ratio 128） | `2·1·head_dim` = 1024 值 | 128 | 16 B |

```text
3×1152 + 13×1024 + 12×16 = 16,960 B = 16.56 KB/token（bf16）
                                    =  8.28 KB/token（fp8）
```

对照与绝对量：

| 模型 | KV/token | 4K | 32K | 128K |
|---|---:|---:|---:|---:|
| Qwen3-8B 全注意力 bf16 | 147.5 KB | 590 MB | 4.7 GB | 18.9 GB |
| MLA 基线 | 31.1 KB | 124 MB | 996 MB | 4.0 GB |
| **P-15B** | **16.56 KB** | **66 MB** | **530 MB** | **2.12 GB** |

> ⚠️ **这是解析值（设计下界），不是任何 serving 栈的实测。** 同一套层结构在
> vLLM 0.29 上实测 **98 KB/token**（12 层混合配置），原因是 compressor 状态缓存
> 用 fp32 存、且 `tokens_per_state` 没传给 spec；其中 **1.40×** 可自己修回来
> （见 [DSV4_10-30B跨卡设计.md](DSV4_10-30B跨卡设计.md) §3.1–3.1.2）。
> **任何容量规划都要写明用的是哪一版口径。**

## 6. 与官方实现的差异（有意为之）

| 项 | 官方 | P-15B | 为什么 |
|---|---|---|---|
| mHC 超连接 | 有（`hc_mult=4`） | **没有** | 只值 0.012% 参数 / 0.3% FLOPs，却把官方路径锁死在 sm90+ |
| hash routing | 前几层有 | 没有 | 训练技巧，与 KV 设计无关 |
| shared expert | 有 | **没有** | 简化；加回是 +9.83M/层 = +0.275B 总参、激活 +18% |
| indexer | 多头 | **单头** | 参考实现的选择：固定的是 top-k 行为，不是头数 |
| fp8 权重 / KV | 硬要求 | 默认 bf16，fp8 可选 | sm80/sm86 没有 fp8 GEMM |
| 压缩层前向 | FlashMLA / FlashInfer-sparse | 纯 torch，可换自研 Triton | 前者只有 Hopper+ |

**踩过并修掉的两个实现错误**（写在这里免得重犯）：

1. **compressor 不能对整窗做 dense 投影**（`hidden×window×state`），HCA 会每层
   134M 参数；改成"窗口池化 + `hidden→state` 小投影"。
2. **压缩层必须做联合 softmax**：窗口 logits 与选中状态 logits 拼接后**一次** softmax。
   分别 softmax 再相加等于给两条支路各 1 的权重，是语义错误（修之前 torch 路径与
   自研 kernel 差 1.1 个 logit）。

## 7. 部署

| 卡 | 方案 | 权重/卡 | 备注 |
|---|---|---|---|
| A100 80GB | **单卡 bf16** | 28.75 GB | 最省事 |
| 5090 32GB | **TP=2 bf16**（或 fp8 单卡） | 14.38 GB | bf16 单卡只剩 ~3 GB，KV 一开就满 |
| 4090 24GB | **TP=2 bf16** | 14.38 GB | 24GB 卡唯一稳妥解 |
| 3090 24GB | **TP=2 bf16** | 14.38 GB | 3090 无 fp8 GEMM |

**为什么不是 fp8 单卡**：3090（sm86）/A100（sm80）上 PyTorch 的 fp8 GEMM 路径不存在
（实测 `torch._scaled_mm` 报 `only supported on CUDA devices with compute capability
>= 9.0 or 8.9`；A100 硬件有 FP8 tensor core，但这条路径不给 sm80）。要用 fp8 省
显存，必须自己写"fp8 常驻 + 逐层 dequant"的加载器。

四类卡的"跑得对"已经验证：同一份参考实现在 **sm80/86/89/120** 上 fp32 logits 指纹
相对差 ≤ 6.3e-7；自研的 sparse-state 注意力与 inv-RoPE 两个 kernel 也都在四类卡上
跑通（`design/dsv4_ref/triton_*.py`）。

## 8. 训练

**预算**（`6 × 激活参数 × tokens`，单卡按 100 TFLOPs 有效值）：

```text
100B tokens: 6 × 1.655e9 × 1e11 / 1e14 =  114.9 GPU-days → 8 卡 14.4 天
1T   tokens:                            = 1149   GPU-days → 8 卡 143.6 天
```

**稳定性**（没有 mHC 之后）：SwiGLU 输出 clamp 在 10.0（官方 `swiglu_limit`，
**已实现**）+ 保守初始化/LR/warmup + 先短上下文。注意 **anticipatory routing
目前只在文档里、代码里没有**，要用得先写。

**路径建议**：不要从零训到 1T。①先用 P-15B 的 config 做**模拟研究**（不需要训练）；
②若要真模型，先用 **P-5B + 100B tokens（8 卡 6 天）** 验证训练与推理管线，再放大。

## 9. 复现与接口

```bash
cd LLMServingSim
# 规模 / KV（不需要 GPU）
docker run --rm --entrypoint python3 -v "$PWD":/work -w /work \
  vllm/vllm-openai:casr029 -m design.dsv4_ref.verify --config p15b --summary-only
# 跨卡指纹（fp32；--kernel triton 换成自研 kernel）
docker run --rm --gpus '"device=0"' --entrypoint python3 -v "$PWD":/work -w /work \
  vllm/vllm-openai:casr029 -m design.dsv4_ref.verify --config p15b \
  --tokens 128 --device cuda --dtype float32 [--kernel triton]
```

**接模拟器的两个接口**：

| 用途 | 位置 | 值 |
|---|---|---|
| 逐层压缩比记账 | `configs/model/DeepSeek/DSV4-P15B-draft.json` 的 `kv_layout` | 28 项 ratio 列表 |
| 链路预算里的 KV 体积 | cluster config 的 `casr.kv_bytes_per_token` | **16960**（= 16.56 KB） |

> ⚠️ draft config 里写的是 `profile_model: Qwen/Qwen3-8B`——**算力借 Qwen3-8B 的
> kernel profile，只有 KV 几何来自本设计**。P-15B 是 **1.655B 激活**的 MoE，prefill
> 的算力只有稠密 8.19B 的 **1/4.95**，所以在"跨域 vs 就近"这类实验里，**这个借用会
> 系统性高估跨域的价值**。以"本域是 3090、远端有 5090"为例（跨域收益 = 算力差 −
> KV×链路差）：
>
> | prompt | 借来的稠密 8B：算力差 / 翻转带宽 | 换成 MoE 算力比：算力差 / 翻转带宽 |
> |---|---|---|
> | 1024 tok | 180 ms / **86 MB/s**（区间之下→总该跨） | 36 ms / **318 MB/s** |
> | 3800 tok | 362 ms / **148 MB/s**（区间之内→会翻） | 73 ms / **462 MB/s**（区间之上） |
>
> 也就是说：用借来的 profile，100–400 MB/s 里存在真实的决策翻转；换成真实 MoE 算力比，
> **整段区间都变成"别跨"**——结论会整个反过来。
>
> ✅ **这条已经修了（2026-09-22）**：路径 B 收完了 P-15B 自己的 profile——
> `profiler/models/p15b.yaml` + 四类卡的 bundle（`profiler/perf/{RTX3090,RTX4090,RTX5090,A100}/casr/P15B/bf16`），
> 上面的"借来的稠密 8B"数字**不再适用于 P-15B 的实验**；凡引用该警告的段落
> 都要改成用真实 bundle 重算。过程见
> [P15B-profile路径B实现计划.md](P15B-profile路径B实现计划.md)。

## 10. 未决

1. **算力 profile 缺失**（上面那条），影响所有 prefill 相关的结论，且影响结论符号。
2. **词表**：65536 vs 官方 129280。换大的要重训 tokenizer。
3. **长上下文**：现在 `max_position_embeddings=4096`；到 32K/128K 要验 YaRN 外推
   （官方 `rope_scaling` 是 yarn factor 16）。
4. **TP=2 通信**：MLA latent 是单头，切分要注意 KV 不切、按 head 切 attention 与 MoE；
   需要一次 TP=2 对照实验确认精度/性能。
5. **shared expert 要不要加回**（+0.275B 总参 / 激活 +18%）。
6. **压缩层 KV 在真实栈上的口径**：解析 16.56 KB vs vLLM 实测 ~98 KB，差一个数量级，
   容量结论依赖哪一版必须写死。
