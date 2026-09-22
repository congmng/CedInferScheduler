# 路径 B 实现计划：把 P-15B 做成可 profile 的 vLLM 模型（2026-09-22）

> **目标**：让 `profiler` 能在 **3090 / 4090 / 5090 / A100** 四类卡上直接量
> [P-15B](DSV4-P15B设计.md) 的逐层成本，产出可被模拟器消费的 bundle。
>
> **为什么必须走 B**：官方 DSV4 路径在四类卡里只有 5090 能起——mHC 的 deep_gemm
> kernel 只编了 `sm90/sm100/sm120`（4090 上实测报 `hyperconnection.hpp: Unsupported
> architecture`），o_proj 又是 fp8 DeepGEMM 专用（无 fp8 GEMM 的卡直接
> `AttributeError: ... weight_scale_inv`）。这不是机器占用问题，是模型本身起不来。

---

## 1. 总体思路

**不去改官方实现，自己写一个 vLLM 原生的 P-15B 模型类。** 三个关键事实让这条路很便宜：

1. profiler 用 **`load_format: dummy`** —— 全程不读权重，只要模块**形状**对。
   **不需要 checkpoint，不需要训练。**
2. profiler 的架构 YAML 里 `vllm:` 字段只是**类名字符串**（`timings.py` 的匹配规则是
   "节点类名 == entry.vllm 且祖先里有 entry.within"），**不要求是 vLLM 自带的类**。
   所以我们的自研模块可以直接写进 catalog。
3. 我们的模型是**标准 pre-norm 残差 + 普通 linear + 自己的稀疏注意力**，
   全是我们已经写过的算子（`design/dsv4_ref/` + `triton_*.py`）。

于是交付物是"一个模型包 + 一份 YAML + 三份 config + 一个注册钩子"。

## 2. 交付物（文件级）

```text
LLMServingSim/
├── deploy/vllm_p15b/                 # 新增：vLLM 原生模型包
│   ├── __init__.py                   # 暴露 P15BForCausalLM + register()
│   ├── model.py                      # 主干：embedding/28 层/head
│   ├── attention.py                  # MLA + 滑窗 + 压缩状态（调用 compressor/indexer）
│   ├── compressor.py                 # 窗口池化 + hidden→state 投影 + RMSNorm
│   ├── indexer.py                    # 打分 + top-k
│   ├── sparse_attn.py                # 稀疏注意力（torch 版 + Triton 版双实现）
│   ├── moe.py                        # gate + 48 专家 top-3 + SwiGLU clamp
│   ├── kv_spec.py                    # get_kv_cache_spec（见 §4.3）
│   └── sitecustomize.py              # 解释器启动时注册模型（挂 PYTHONPATH 即可）
├── configs/model/casr/
│   ├── P15B-r0.json                  # 1 层，layers_block_type=[0]     （全 MLA）
│   ├── P15B-r4.json                  # 1 层，layers_block_type=[4]     （CSA）
│   └── P15B-r128.json                # 1 层，layers_block_type=[128]   （HCA）
└── profiler/models/p15b.yaml         # catalog + sequence + layer_types
```

## 3. 实施步骤（每步一个判据）

### 步骤 1：形状对齐（先做，最容易发现接错）

把 [P-15B 设计](DSV4-P15B设计.md) §2/§4 的字段写进三份 `P15B-r*.json`：`hidden_size=2560`、
`num_attention_heads=20`、`head_dim=512`、`qk_rope_head_dim=64`、`q_lora_rank=640`、
`o_lora_rank=512`、`o_groups=8`、`num_experts=48`、`num_experts_per_tok=3`、
`moe_intermediate_size=1280`、`vocab_size=65536`，外加 `intermediate_size`（=1280，
见 §4.2）和 `layers_block_type`。

**判据**：`LLM(model=<config>, load_format="dummy", ...)` 能起来并 forward 一次，
且**单层解码器的参数量**（把 embedding / lm_head / final-norm 扣掉）等于
`design/dsv4_ref` 用同一份单层配置建出来的值——全 MLA 层 **497,383,616**、
CSA 层 **503,365,824**、HCA 层 **500,415,680**（这三个数已与 28 层总量对账过，
见 [P-15B 设计](DSV4-P15B设计.md) §4）。**这条是防接错的主闸门。**

### 步骤 2：四卡跑通（这是 B 的意义所在）

在本机 4090、3090a、5090、A100-80G 上各起一次 dummy 模型 + forward。

**判据**：四类卡都无异常退出；4090/3090/A100 上**不再出现 mHC 或 fp8 相关的报错**。

### 步骤 3：写 `profiler/models/p15b.yaml`

catalog 的 `vllm:` 直接写我们的类名；`attention:` 只能有 **1 个** entry（schema 强制）；
`layer_types:` 写 `r0 / r4 / r128` 三条流水线给模拟器用。

**判据**：`python3 -m profiler profile --help` 不报 schema 错；写错的层名在 profile
**之前**就报（`extra="forbid"` + `_check_catalog`）。

### 步骤 4：三份 config × 四张卡 = 12 次 profile

**为什么是 3 份而不是 1 份**（读代码才发现的坑）：profiler 固定用
`hf_overrides: {num_hidden_layers: 1}` profile **一层**，而 P-15B 的三种层
**形状不同**（compressor 2048 / 1024 / 无）。Zamba2 那种"一层里同时有 mamba 和
attention"的结构可以一次量完，**我们的层类型是跨层分布的，做不到**。

所以每次只 profile 一种层类型；三种类型的 canonical 层名**故意取不同的名字**
（`compressor_r4` / `compressor_r128` / `kv_state_proj_r4` …），这样三份 CSV 的
行键不冲突，可以用现成的 `tests/merge_profile_shards.py` 按"互斥且全覆盖"合并成一个
bundle（它本来就是按 category 主键断言并集的）。注意 `attention:` 只能有一个 canonical
名，三种类型共用它——因为注意力 kernel 的查表键是
`(prefill_chunk, kv_prefill, n_decode, kv_decode)`，不含形状。

**判据**：四类卡各产出一个 bundle，`tests/check_profile_bundle.py` 通过，
`tests/check_profile_bundle_consistency.py` 全 `ok`。

### 步骤 5：接进模拟器

`p15b.yaml` 的 `layer_types` 让 `trace_generator._block_type_for` 按
`layers_block_type[i]` 选流水线（Zamba2 那轮已经实现并测试过）；
cluster config 里的 `casr.kv_bytes_per_token` 设成 **16960**。

**判据**：一次模拟跑完，逐层成本非零且与 bundle 的层名对得上（没有 `Layer ... missing
from the profile CSVs` 的 warning）。

### 步骤 6（可选，性能版）：把 `sparse_attn.py` 换成 Triton

`design/dsv4_ref/triton_sparse_attn.py` 已经在 sm80/86/89/120 上跑通并对过 fp64。
把它接进来再 profile 一次，得到"接近最终实现"的注意力成本。

**判据**：换前/换后两次 bundle 都在，且 Triton 版的 attention.csv 更快；差异写进文档。

## 4. 四个关键技术决定

### 4.1 用 vLLM 的并行层，而不是自己写切分

`QKVParallelLinear` / `RowParallelLinear` / `ColumnParallelLinear` / `RMSNorm` /
`VocabParallelEmbedding` / `LogitsProcessor` / `RotaryEmbedding` 直接复用。好处：
① profiler 的 canonical 名字大多现成可用；② TP 切分由 vLLM 负责，
profiler 的 `--tp 2` 模拟（拿 `HF_OVERRIDES` 缩小形状）自动成立。

### 4.2 `intermediate_size` 必须设成专家宽度

profiler 的 TP 模拟只切四个字段：
`SHARD_FIELDS = [intermediate_size, num_attention_heads, num_key_value_heads, vocab_size]`。
**`moe_intermediate_size` 不在里面**。所以 config 里要把 `intermediate_size` 也写成
1280（= 专家宽度），否则 TP=2 的专家形状不会被切分，profile 出来的 MoE 是错的。
（Qwen3-30B-A3B 的 bundle 里记的 `intermediate_size: 768` 就是这个道理。）

### 4.3 KV spec 我们自己给，而且给对

我们的 KV 是"全层 latent + 每 ratio 个 token 一个压缩状态"，不是标准 paged KV。
模型要实现 `get_kv_cache_spec`，**并且从一开始就写对**：

```python
SlidingWindowMLASpec(block_size=ratio, tokens_per_state=ratio, ...)   # 压缩层
MLAAttentionSpec(tokens_per_state=1, ...)                             # 全 MLA 层
```

这正是我们在官方实现里查出来的那个缺陷（`compressor` 漏传 `tokens_per_state` 导致
一层白吃 4–8 KB/token）。自己写就不会带上它——**顺带把 §3.1.2 里"修好账能拿回
1.40× 容量"这件事在我们的模型上默认兑现**。

### 4.4 profile 出来的是"可移植实现"，不是"优化后的 kernel"

第一版注意力用 torch（banded softmax + gather），Moe 用循环或 vLLM 的 fused MoE。
所以第一版 bundle 的**绝对时间偏保守**，尤其是注意力与 MoE 的算子效率。文档里要标明，
否则模拟器会拿"未优化实现"当"最终性能"用。步骤 6 提供修订版。

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| 模型写错形状，profile 出一堆看似合理的假数 | 步骤 1 的参数量闸门（逐层类型对账到个位） |
| 1 层模型只量到一种层类型（§4 的坑） | 三份 config + 按不同 canonical 名合并 |
| profiler 在 `num_hidden_layers=1` 下要求 KV cache 合法 | §4.3 的 spec；先在 4090 上验证 |
| TP=2 时 MLA 单头 latent 的切分 | KV 不切；Step 2 里 `--tp 1,2` 都跑一遍 |
| `Sitecustomize` 注册模型在 worker 进程不生效 | 走 `PYTHONPATH` 的 `sitecustomize`（每个新解释器都会加载），先在一个卡上验证 |
| 我们的 MoE 与 vLLM fused MoE 的类名/形状不一致 | MoE 直接用 vLLM 的 `FusedMoE`，catalog 绑定它的类名 |
| 12 次 profile 的时间 | 单次与 Qwen3-8B 同量级（`num_hidden_layers=1`，MoE 稍大）；按卡分片并发 |

## 6. 资源与时间估算

- **卡**：现在就有 4090×2（本机）、3090×2（3090a）、5090×2、A100-80G×4（0–3 空）。
- **单次 profile**：与 09-18 那次同量级（30–40 min，`iters 3`、`attention_max_kv 16384`）。
- **总量**：3 类型 × 4 卡 = 12 次；按卡并发、类型串行 → 每卡 3 次 ≈ **2 小时**（A100 四卡可再分片压缩）。
- **不需要**：checkpoint、训练、官方镜像里的 DSV4 代码路径。

## 7. 与现有资产的复用

| 复用 | 来源 |
|---|---|
| 模型结构、压缩/滑窗/top-k 语义 | `design/dsv4_ref/model.py` |
| Triton 稀疏注意力（已验证四卡） | `design/dsv4_ref/triton_sparse_attn.py` |
| Triton inv-RoPE（已验证四卡） | `design/dsv4_ref/triton_o_proj.py` |
| KV spec 的正确写法 | `design/dsv4_spec_fix/`（官方版的反例） |
| 分片合并 / bundle 校验 / 口径一致性 | `tests/merge_profile_shards.py`、`check_profile_bundle.py`、`check_profile_bundle_consistency.py` |
| 逐层流水线（多 block type） | `profiler/models/zamba2.yaml` 的 `layer_types` + `trace_generator._block_type_for` |
