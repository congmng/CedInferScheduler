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

配套的闸门与工具（`tests/`）：

| 文件 | 作用 | 何时跑 |
|---|---|---|
| `check_p15b_shapes.py` | 参数量三方对账 + 与 `design/dsv4_ref` 的 logits parity | 改模型后 |
| `check_p15b_boot.py` | 四类卡装载 + KV 账目闭式 | 改 KV spec 后 |
| `check_p15b_catalog_binding.py` | 每个 canonical 名必须**唯一**绑定一个模块（新增，见"第四道缺口"） | **每次采集前** |
| `test_head_binding.py` | 所有 yaml 的 `lm_head` / `sampler` 必须绑到引擎真正调用的模块（torch-free） | 改 yaml 后 |
| `run_p15b_profile.sh` | 三类 × 一卡；`BLOCKS` / `CATEGORIES` 选子集 | 采集 |
| `merge_profile_types.py` | 三份单类型 bundle 合一 + 共享层一致性闸门 | 采集后 |
| `assemble_p15b_bundle.py` | 把"正式采集 + dense 重采"拼成一个 type-root，再调合并与两个校验器 | 一个域收尾时（一条命令） |

## 3. 实施步骤（每步一个判据）

### 步骤 1：形状对齐 ✅ **已完成（2026-09-22）**

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

**实测结果**（`tests/check_p15b_shapes.py`，三条互相独立的表述必须一致）：

```text
r0    ratio=0    ours=497,383,616 reference=497,383,616 expected=497,383,616 ok
r4    ratio=4    ours=503,365,824 reference=503,365,824 expected=503,365,824 ok
r128  ratio=128  ours=500,415,680 reference=500,415,680 expected=500,415,680 ok
```

另外补了一条比参数量更硬的检查：**把参考实现的权重按名字映射过来，比较两者
logits**（fp32、T=256，三种层类型）——结果 **max|Δlogits| = 0.000e+00**，
即这份 vLLM 形状的模块树与参考实现在数值上完全一致。前向本身也跑了
（T=32/256，三种类型全部 finite）。

过程中修掉三处：`state_values` 多套了一层 `[:, None]`（4D/5D 不匹配）、
`state_pos` 的 valid 掩码多套一层（广播成 `(b,b,t,K)`）、以及闸门自己把参考模型的
embedding 名（`embed` vs `embed_tokens`）漏扣了 167,772,160 个参数。

### 步骤 2：四卡跑通 ✅ **已完成（2026-09-22）**

在本机 4090、3090a、5090、A100-80G 上各起一次 dummy 模型 + forward。

**判据**：四类卡都无异常退出；4090/3090/A100 上**不再出现 mHC 或 fp8 相关的报错**。

**实测（`tests/check_p15b_boot.py`，`load_format=dummy` + `generate` 2 token）**：

| 卡 | 架构 | r0（设计 1152 B/tok） | r4（设计 1024） | r128（设计 16） |
|---|---|---:|---:|---:|
| RTX 3090 | sm86 | **1151 ok** | **1023 ok** | **16 ok** |
| RTX 4090 | sm89 | **1152 ok** | **1024 ok** | **16 ok** |
| RTX 5090 | sm120 | **1152 ok** | **1024 ok** | **16 ok** |
| A100-SXM4-80GB | sm80 | **1152 ok** | **1024 ok** | **16 ok** |

四类卡**都装载成功、都能 `generate`、KV 账目都与设计闭式一致**。这正是路径 B 的
全部意义：官方实现只在这四张卡里的一张（5090）上能实例化，我们这份在四张上都能。

**当前实测（2026-09-22，本机 4090）**：vLLM 装载已经过掉五道关卡，卡在第六道——

| # | 关卡 | 处理 |
|---|---|---|
| 1 | `model_type: p15b` 不被 Transformers 认识 | `hf_config.py` 用 `AutoConfig.register` 注册（官方扩展点） |
| 2 | 模型目录必须含 `config.json` | 与 profiler 一样，把 config 拷进临时目录 |
| 3 | registry 里没有这个架构 | `sitecustomize` 里 `ModelRegistry.register_model` |
| 4 | `VllmModel` 协议要求 `embed_input_ids` | 补上；同时去掉 `SupportsPP`（会要求 `intermediate_tensors`） |
| 5 | runner 传**扁平**布局 `(num_tokens,)`，且带 `inputs_embeds` | 包装层 reshape 成 `(1,T)` 再摊平回来（见下） |
| 6 | `HybridKVCacheCoordinator requires at least one cacheable group` | **已解决**：每层挂一个 `P15BCache`（`deploy/vllm_p15b/kv_cache.py`），声明 spec + `bind_kv_cache`，attention 自己不算 cache 的持有者 |

**第六道关解决后的实测（本机 4090，2026-09-22）**：

```text
r0    BOOT OK   kv=6.30 GiB  cap=5,871,840 tok   -> 1153 B/token   （设计闭式 1152）
r4    BOOT OK   kv=4.23 GiB  cap=4,436,832 tok   -> 1024 B/token   （设计闭式 1024）
r128  BOOT OK   kv=6.27 GiB  cap=420,784,256 tok ->   16 B/token   （设计闭式   16）
```

三种层类型的 **KV 账目与设计闭式逐位吻合**——这正是官方实现算错的那笔账（它的
compressor 少了 `tokens_per_state`，同样的层会记成 8,208 / 4,104 B/token）。
另外整条链路也通了：`llm.generate(...)` 用 dummy 权重跑出 4 个 token
（`GENERATE OK produced 4 tokens: [11196, 15, 11196, 15]`），
`tests/check_p15b_shapes.py` 的参数量与 parity（1e-6）都仍然通过。

**还差的一步（诚实标注）**：现在 attention 仍然是**在 batch 自带的 latent 上**做窗口
与 top-k 的联合 softmax，**没有去读那个 paged cache**。所以引擎起得来、cache 也分配了，
但 profiler 的 attention 扫描里 **kv 那一维不会让时间变化**——`P15BCacheMetadataBuilder`
（block_table / slot_mapping）已经写好，但还没有被 attention 消费。这是 §4.3bis 里 C1
的剩余部分；C2（用现成 MLA 层做代理）仍是时间紧时的退路。

### 4.3bis 新发现：attention 的 profile 与 KV cache 机制是耦合的

第 6 道关卡不是"补一个 spec 就完"的问题。profiler 的 attention 类别是按
`(prefill_chunk, kv_prefill, n_decode, kv_decode)` 打点的——**它假设模型的
attention 会真的去读一个由这些长度决定的 paged KV cache**。而我们第一版的注意力是
纯 torch、在**当前 batch 的 latent** 上做窗口+选中状态的软最大化，根本不碰 cache。
两条后果：

1. engine 侧直接报 `requires at least one cacheable group`（没有 cacheable 的层）；
2. 即使补上一个 spec 让引擎起得来，测出来的 attention 曲线**不会随 kv 长度变化**，
   而模拟器的 attention 模型恰恰是建立在"随 kv 变化"上的——那张表会变成常数，
   等于白测。

所以 attention 那一段必须**真的走 paged cache**。三条路：

| 路 | 做法 | 代价 | 结果可信度 |
|---|---|---|---|
| **C1** | 写一个 vLLM `AttentionImpl` + 自定义 backend，按我们的窗口+top-k 稀疏模式从 paged cache 读 | 中高（要接 backend 注册与 cache 接口） | 最真实：形状、稀疏模式、cache 访问都是我们的 |
| **C2** | 用 vLLM 现成的 **MLA attention 层**（如 DeepSeek-V2 的 `MLAAttention`）做 attention 的 profile 代理，我们自己的稀疏算子留在模型里 | 低 | 是**代理**：cache 机制真实、但读的是稠密 MLA 而不是 top-k 稀疏，需要显式标注 |
| **C3** | 只给一个 spec 让引擎能起，attention 表按 C2 或单独的实验补 | 最低 | 表的 kv 维度不可信 |

**建议 C1**，理由是它顺带把我们自己那条稀疏路径的 cache 语义（`tokens_per_state=ratio`
那笔账）在引擎里跑通——这正好是官方实现出错的地方，自己写一遍就把 §3.1.2 的
1.40× 容量收益从"应该能拿回"变成"实现里本来就对"。若时间紧，先 C2 出一版 bundle
并把"attention 是稠密 MLA 代理"写进 meta，再回头做 C1。

### 步骤 3：写 `profiler/models/p15b.yaml` ✅ **已完成（2026-09-22）**

catalog 的 `vllm:` 直接写我们的类名；`attention:` 只能有 **1 个** entry（schema 强制）；
`layer_types:` 写 `r0 / r4 / r128` 三条流水线给模拟器用。

**判据**：`python3 -m profiler profile --help` 不报 schema 错；写错的层名在 profile
**之前**就报（`extra="forbid"` + `_check_catalog`）。

**实测**：`profiler/models/p15b.yaml` 过 pydantic 校验（21 个 catalog 条目、
3 条 `layer_types` 流水线），`profiler/core/config.py` 的 `_check_catalog` 全过。
为满足"每个 canonical 名必须对应唯一的 `(类名, 父类名)`"：

* 每个 RMSNorm 与每个 linear 都是**独立的类**（`P15BQNorm` vs `P15BKVNorm`、
  `P15BQKVDown` vs `P15BQUp`），否则同一父模块下两个同类会歧义；
* 压缩侧再按 ratio 拆成 `P15BCompressorCSA/HCA`、`P15BIndexerCSA/HCA`、
  `P15BKVStateProjCSA/HCA`——**这样三份单类型 profile 的行键才不冲突**，
  可以合并进同一个 bundle（步骤 4 的前提）；
* 新增 `P15BRotary`：RoPE 原本是函数，而 catalog 只能绑**模块**，不包一层的话
  这一项在逐层成本里会被无声漏掉（无参数，不影响参数量与 parity）。

### 步骤 4：三份 config × 四张卡 = 12 次 profile（**试跑已跑完，剩 attention 一处**）

**试跑实测（本机 4090，小网格：`msq 16 / mnbt 512 / max_kv 512 / iters 1`）**：
模型能被 profiler 正常装载、dense 与 attention 两类都能采集，但在 **moe 类别**报错：

```text
Exception: Call to collective_rpc method failed:
Expected exactly one FusedMoE layer in the test model, got 0
```

也就是说 profiler 的 moe 类别是**按 vLLM 的 `FusedMoE` 写的**——它要通过
collective_rpc 去逐专家驱动一次。我们的 `P15BMoE` 是手写的 48 专家 python 循环，
profiler 认不出来。**这正好也是我们本来就该改的**：MoE 占 91.9% 的参数，手写循环
既不是真实 kernel，也会把 MoE 时间系统性抬高（正是 3090 那种"带宽只跑到 38%"
的同类错误）。V-15B 的正确接法是照 `Qwen3MoeSparseMoeBlock`：
`ReplicatedLinear` gate + `FusedMoEFactory(shared_experts=None, gate=..., top_k=3,
intermediate_size=1280, prefix=...)`，forward 里
`self.experts(hidden_states=..., router_logits=...)`。

**注意权重布局不同**：参考实现用 `up (E, H, 2*I)` / `down (E, I, H)`，
而 `FusedMoE` 存 `w13_weight (E, 2*I, H)` / `w2_weight (E, H, I)` —— **是转置过的**，
所以 `tests/check_p15b_shapes.py` 的 parity 映射要跟着加转置（参数量不变）。

**第二处**是 §4.3bis 的遗留：attention 还是读 batch 自带 latent、不读 paged cache，
所以哪怕跑完，attention 表在 kv 那一维也是平的。

**① 已完成（2026-09-22）**：`P15BMoE` 现在有两条后端、共用一个类名——
`vllm_config is None` 时走纯 torch（步骤 1 的参数量/parity 闸门用它），
否则走 `ReplicatedLinear` gate + `FusedMoEFactory`（照 `Qwen3MoeSparseMoeBlock`）。
换完之后**四类采集全部跑通**：

```text
TP=1 dense        56/56    ✓ dense.csv
TP=1 per_sequence 16/16    ✓ per_sequence.csv
TP=1 attention  1080/1080  ✓ attention.csv
TP=1 moe          30/30    ✓ moe.csv
Results written to: /out/P15BTEST/casr/P15B-r4/bf16
```

dense.csv 里 **14 个 canonical 层名全部出现**（`qkv_down` / `q_up` / `o_lora_a` /
`o_lora_b` / `compressor_csa` / `kvproj_csa` / `indexer_csa` / `rotary_emb` /
`input_norm` / `ffn_norm` / `final_norm` / `embedding` …），说明 catalog 绑定正确；
`moe.csv` 30 行。旁证：参数量与 parity 闸门在换 MoE 后**仍然通过**（权重布局从
`up (E,H,2I)` 变成 `w13 (E,2I,H)` 是转置关系，参数量不变，而纯 torch 那条路没动）。

**② 还没做，而且现在有实测证据了**：attention 表在 kv 维度上**确实是平的**——

```text
prefill_chunk=128, n_decode=1:
   kv_decode=16    time_us=79.9
   kv_decode=64    time_us=79.8
   kv_decode=512   time_us=80.0        # 32× 的 kv，差 0.2%
```

即"引擎起得来、cache 也分配了，但 attention 不读它"。模拟器的 attention 模型正是
建立在"随 kv 变化"上的，所以这张表**目前不可用**（dense 与 moe 两类是真实可用的）。
下一步二选一：**C1** 让 attention 真读 paged cache（`P15BCacheMetadataBuilder` 的
block_table / slot_mapping 已就绪，缺的是在 attention 里消费它），或 **C2** 用现成
MLA 层做 attention 代理并在 meta 里显式标注。

#### C1 完成（2026-09-22）：attention 真的读 paged cache 了

（这一节前面曾写过一个**错误结论**——"C1 要动 runner"——原因是探针**只报第一次调用**，
而第一次是 warmup，`attn_metadata` 本来就没有。探针改成按状态上报后真相是：）

```text
[p15b.meta] p15b.layers.0.attn: attn_metadata=NoneType                      ← warmup
[p15b.meta] p15b.layers.0.attn: attn_metadata keys=['p15b.layers.0.kv_cache'] (present: True)
[p15b.meta]   P15BCacheMetadata: block_table=(4, 64) slot_mapping=(8,) block_size=4
```

runner **确实**给我们的层建了 metadata，用的就是我们的 `P15BCacheMetadataBuilder`——
只是**记在 cache 模块的 prefix（`...kv_cache`）下**，而 attention 一开始用自己的
prefix 去查。修三处就通了：

1. 按 `f"{prefix}.kv_cache"` 取 metadata（runner 是按 KV cache group 命名的）；
2. `block_table` 是**按 request** 分配的（`max_num_reqs` 行），而 attention 是单序列
   batch，只取前 `B` 行；
3. 表被 pad 到 `max_blocks`，整张读下来大小恒定（kv 轴照样是平的）——
   要用 `seq_lens` 截到真实缓存长度，压缩层再除以 `ratio` 才得到状态数。

**实测效果（同一网格，改前 vs 改后）**：

| `prefill_chunk` / `n_decode` | 改前 kv 16→512 | 改后 kv 16→512 |
|---|---|---|
| 128 / 1 | 79.9 → 80.0 µs（0.2%） | **106.4 → 162.9 µs（1.53×）** |
| 256 / 1 | — | **190.2 → 305.5 µs（1.61×）** |
| 128 / 4 | — | **124.4 → 186.5 µs（1.50×）** |

attention 那一列**现在随 kv 变化**，模拟器的 attention 模型可以用它。
（C2 那条"用 MLA 代理"的退路不再需要。）

**一条保真度警告（必须跟着 bundle 走）**：现在实现的是**读路径**——attention 按
`block_table`/`slot_mapping` 从 paged cache 里取行并参与联合 softmax，所以**成本**方向
是对的。但**写路径没有实现**：模型不会把自己算出来的状态（`compress → norm → store`）
写进 cache，所以 attention 读到的数值不是这个模型自己的状态。对 profiler 的用途
（量成本随 kv 的变化）没有影响，但**这份 bundle 的 attention 列只能当成本模型用，
不能用来判断数值正确性**。要补写路径，参考官方 `Compressor` 的
`compress_norm_rope_store`（Triton kernel）。

### 步骤 4 的正式采集：已启动（2026-09-22）

启动脚本 `tests/run_p15b_profile.sh <hardware> <out-root> [shard_i shard_n]`，
对每种 block type 各跑一次，用与 09-18 重采相同口径
（`tp 1`、`msq 128`、`mnbt 2048`、`max_kv 16384`、`iters 3`、`--skip-skew`）：

| 域 | 主机 | 镜像 | 产出目录 |
|---|---|---|---|
| RTX4090 | 本机 GPU0 | `casr029` | `/tmp/p15b-formal/{r0,r4,r128}/` |
| RTX3090 | `10.212.67.68` GPU0 | `casr029` | 同上（该机 `/tmp/p15b-formal`） |
| RTX5090 | `10.212.70.196` GPU2 | `casr029` | 同上 |
| A100 | `10.70.251.47:2222` GPU0 | `v0.29.0` | `~/p15b-out/{r0,r4,r128}/` |

三种类型的产出**分三个目录**（因为各自是一次独立 run），合并时按 canonical 名取并集——
层名已按类型区分（`compressor_csa` / `compressor_hca` …），所以并集是互斥的。

**耗时修正（实测）**：原本估"单域 2 小时"，实际 **attention 阶段是绝对瓶颈**——
`max_kv 16384` 下每类 7000+ 个 shot，而 P-15B 的 attention 每次要做
"gather 缓存 + 窗口 + top-k 联合 softmax"，比 Qwen3-8B 的注意力重得多：
**r0 一类跑了 40+ 分钟还没完**。单域 3 类串行，实际更接近 **4–5 小时**
（`dense` 与 `per_sequence` 各自只要 2 分钟上下）。

**并行化（2026-09-22）**：每台机器都有第二张空卡，所以把 GPU0 会**最后**做的那一类
（r128）挪到第二张卡上同时跑——`tests/run_p15b_profile.sh` 加了 `BLOCKS` 环境变量，
于是每台机器上是：

```text
GPU0：r0 → r4 → r128（脚本默认的串行波次）
GPU1：r128 单独一次                  （BLOCKS=r128）
```

波次完成时间从 `r0+r4+r128` 降到 `r0+r4`，每域省约 1 小时。r128 会被测两遍
（GPU0 那遍是重复的）——**重复不浪费**：合并工具对重复行做的一致性校验正好多一个样本，
两边不一致就说明跑本身有问题。GPU0 波次到 r4 落地后可以直接停掉。

#### 合并工具（`tests/merge_profile_types.py`）：已写好并在小网格上验证

三种类型的 CSV 有两类行键：

* **类型专属**（`compressor_csa` / `compressor_hca` / `indexer_*` / `kvproj_*`）——
  直接并集；
* **三类型共享**（`embedding` / `qkv_down` / `q_up` / `o_lora_*` / `moe` / 各 norm）——
  同名同形状，所以**同一次测量会出现三次**。工具保留第一份，并把差异当**免费的一致性
  校验**报出来（小网格 `iters=1` 实测：`dense` 逐层中位偏差 **13.1%**、`per_sequence`
  **0.1%**、`moe` **5.3%**，都在 15% 阈值内；单行最差 27.9%，是小算子的噪声，
  所以判据用**逐层中位**而不是最差单行）。

**attention 不能合并——这是工具抓出来的一个真问题**：同一个 shot 键在三种类型间最大
差 **77.4%**。原因是三种类型的注意力根本不是同一个算子（r0 全因果、r4 窗口 8 + top-k、
r128 窗口 128 + top-k）。而 profiler 的 schema 强制 `catalog.attention` **恰好一项**，
simulator 的 `_lookup_attention` 也只有一个表——**"一个模型一个 attention 表"这个假设
对 P-15B 不成立**。工具因此把三张表并排写进 bundle：

```text
attention.csv        ← r0（全因果）
attention_r4.csv     ← CSA（窗口 8 + top-k）
attention_r128.csv   ← HCA（窗口 128 + top-k）
```

**已完成（2026-09-22）**：`trace_generator` 现在按 `layers_block_type` 选 attention 表——
`_build_tp_tables` 会加载同目录下的 `attention_<block>.csv`，
`_lookup_attention(..., table=...)` / `_lookup_attention_with_skew(..., table=...)`
按类型查，`_emit_layer` 用 `_block_type_for(ctx, layer_num)` 决定用哪张；
**没有这类文件的模型（Qwen3 / Zamba2 等）行为不变**（回落单表）。

用合成三表验证过分派：`attention` 151.2 µs / `attention_r4` 60.5 µs /
`attention_r128` 105.8 µs，`table=None` 时回落到 `attention` ✓。

#### 步骤 4 的第四道缺口：一个类名被两处绑定（2026-09-22，已修）

**怎么发现的**：合并工具的一致性闸门（三份单类型采集里的**共享层**互为重复测量）
在 5090 的三次采集上报了一处异常——11 个共享层里 10 个逐层中位偏差 ≤ 2%，只有
`kv_norm` **17.3%**，超过 15% 阈值，工具**拒绝合并**：

```text
dense.csv: 2516 rows ... worst layer median 17.3%
these shared layers disagree systematically across block types (median > 15%): kv_norm
```

**直接定位**：新增 `tests/check_p15b_catalog_binding.py`——照 profiler 自己的匹配
规则（类名相等 + `within` 出现在祖先类名链里，`profiler/core/hooks/timings.py`）
在模块树上复现一遍，报出每个 canonical 名**命中了几个模块**。跑的第一次就点出来两处：

```text
r4   dense  kv_norm  AMBIGUOUS   layers.0.self_attn.kv_norm            576
                                 layers.0.self_attn.compressor.norm   2048   ^ also
r128 dense  kv_norm  AMBIGUOUS   layers.0.self_attn.kv_norm            576
                                 layers.0.self_attn.compressor.norm   1024   ^ also
r0/r4/r128  dense  o_lora_a AMBIGUOUS   o_lora_a.0 … o_lora_a.7   （8 个）
```

**为什么必须修**（不是"数字难看"，是记账口径错了）：profiler 的 `DedupSink`
对同一个 `(layer, tokens)` 键的多个采样取**平均**（`profiler/core/writer.py`），
而模拟器按名字逐层计价**一次**。于是：

| 行 | 修前的实际含义 | 后果 |
|---|---|---|
| `kv_norm`（r4/r128） | 注意力 576 宽 norm 与 compressor 2048/1024 宽 norm 的**均值** | 与 r0 系统性差 15–40% |
| `o_lora_a`（三类） | **单组**投影的成本（8 组被平均成 1 组） | 模拟器只计一次 → **8 倍低估** |

**修法**（两处，都不动数值语义）：

1. compressor 内部的 norm 换成新类 `P15BCompressorNorm`（**不**进 catalog）。
   compressor 自己的条目是 inclusive 的，本来就把这层 norm 的钱算进去了，所以
   这是"去掉重复计费"而不是"少算一层"。
2. `o_lora_a` 从 `nn.ModuleList` 换成 `P15BOLoRAGroups`（`ModuleList` 子类，
   `forward` 里把 8 组一次算完并 `cat`），catalog 绑**容器**而不是单组——
   容器的节点时间是包含 8 个子模块的（同 `qkv_down`/`o_lora_b` 这些 wrapper
   的行为）。

**实证**（本机 4090，只重采 dense 类，每个类型约 3 分钟）：

```text
o_lora_a   tokens=1     3.57 ->  30.06 µs  (8.42x)
           tokens=512  10.14 ->  85.06 µs  (8.39x)
           tokens=2048 18.64 -> 158.42 µs  (8.50x)
q_up / o_lora_b / kv_norm / 其余每一层      0.99 - 1.01x   （没被动过）

kv_norm 跨类型离散度 (max-min)/max：修复前 中位 15.2% / 最大 40.2%
                                    修复后 中位  2.0% / 最大 13.5%（tokens=1 的 2 µs 小算子）
合并闸门：dense 最差逐层中位偏差 17.3% -> 2.3%   （通过）
```

> **这条不只影响两行数字**：修正后 2048 token 的 `o_lora_a` 是 **158 µs**，与
> `q_up`（165 µs）、`o_lora_b`（249 µs）同量级；修前它看起来是最便宜的注意力行
> （18.6 µs）。逐层成本分布因此变了，凡是拿"哪个阶段便宜"做的判断都要重看。

**增量重采（不必重跑 4 小时的 attention）**：profiler 新增 `--categories`
（`--force` 只重写**运行类别**自己的 CSV，其它文件不碰），启动脚本透传 `CATEGORIES=`：

```bash
# 只修 dense.csv，attention.csv / moe.csv / per_sequence.csv 原样保留
CATEGORIES=dense tests/run_p15b_profile.sh RTX4090 /out

# 一个域收尾：拼接 → 合并 → 校验（校验不过就非零退出）
python3 tests/assemble_p15b_bundle.py --hardware RTX5090 \
    --profile-root /tmp/p15b-formal --dense-root /tmp/p15b-dense \
    --work-root /tmp/p15b-assembled-5090 \
    --note "dense re-measured 2026-09-22 after the catalog binding fix"
```

### 步骤 4 落地状态（2026-09-22 20:20）

#### 步骤 4 的第五道缺口：head 绑到了一个"从不被调用"的类（2026-09-22，已修）

**怎么发现的**：把新 bundle 喂给模拟器，整个跑通、逐层都查得到，**只有一条告警**：

```text
[TraceGenerator] WARNING  Layer 'lm_head' is in the architecture yaml sequence
but missing from the profile CSVs for RTX4090/casr/P15B/bf16 — skipping.
```

head 是 `2560 × 65536 = 167.8M` 参数、约占每 token 激活算力的 **10%**，不能被静默跳过。

**定位**（对照实验）：同一个镜像、同一个类别，给 Qwen3-8B 采一次 per_sequence——
它是 **32 行 `lm_head`、没有 `sampler`**；而 P-15B 是 **32 行 `sampler`、没有 `lm_head`**。
读 vLLM 源码后原因很清楚：`LogitsProcessor._apply_head` 走的是
`lm_head.quant_method.apply(lm_head, hidden_states, ...)`，**从不调用 `ParallelLMHead`
模块本身**，所以 profiler 的"模块调用树"里永远没有这个节点。
而**所有其他架构 yaml 早就把 canonical 名 `lm_head` 绑到 `LogitsProcessor`**
（真正做 head 投影的模块）、`sampler` 绑到 `Sampler`。我们绑成了
`ParallelLMHead` / `LogitsProcessor`——正好错位一格。

**旁证（数值）**：P-15B 那 32 行 `sampler`（其实是 head GEMM）与 Qwen3-8B 的
`lm_head` 行，按 head 形状比 `(2560×65536)/(4096×151936) = 0.2696` 缩放后逐点吻合：

| sequences | P-15B（记为 sampler） | Qwen3-8B lm_head × 0.2696 |
|---:|---:|---:|
| 1 | 360.7 µs | 350.2 µs |
| 16 | 364.0 µs | 359.3 µs |
| 128 | 426.4 µs | 411.0 µs |

**后果**：模拟器查 `lm_head` 查不到（跳过），而 `sampler` 那一行装的其实是 head 的成本——
名字与内容错位，任何"head 贵还是 sampler 贵"的判断都不可信。

**修法**：yaml 两行改成 `lm_head: LogitsProcessor` / `sampler: Sampler`（带 `tp_stable`），
并加 `tests/test_head_binding.py`：**torch-free**，直接读所有 yaml，钉住"head 必须绑到
引擎真正调用的模块"这条不变量（一跑就覆盖全部 5 个 yaml）。重采只需 per_sequence 一个
类别（约 1 分钟/类型）：

```bash
CATEGORIES=per_sequence tests/run_p15b_profile.sh RTX4090 /out
```

| 域 | 正式采集（三类×4 类 CSV） | dense 重采 | per_sequence 重采 | bundle 落地 |
|---|---|---|---|---|
| RTX4090（本机 GPU0/1） | ✅ 齐 | ✅ | ✅ | ✅ `profiler/perf/RTX4090/casr/P15B/bf16`，两个校验器全过 |
| A100-80G | ✅ 齐（r4 于 18:33 收尾） | ✅ | ✅ | ✅ `profiler/perf/A100/...`（moe 合并用 `--max-spread 0.2`，理由写在 meta 的 notes 里） |
| RTX5090 | ✅ 齐（10:01 UTC） | ✅ | r128 进行中 | 待合并 |
| RTX3090（3090a） | ✅ 齐（r4 于 20:10 收尾） | r128 进行中 | r128 进行中 | 待合并 |

`meta.yaml` 现在会带 `notes:`（记录 dense 是哪次重采的）和按**当前** yaml 重算的
`architecture_sha256`——合并把两次采集拼在一起时，这是唯一说得清 provenance 的地方。

**模拟器侧已验证**（单域 1P+1D，用 RTX4090 的真实 bundle）：

```bash
CLUSTER_CONFIG=configs/cluster/casr_p15b_rtx4090_1p1d.json \
  bash tests/run_casr_comparison.sh /tmp/p15b-smoke-4090
```

跑完 8 个请求、逐层都能查到成本；修 head 绑定之前它只报 `lm_head` 缺失，现在是
唯一剩下的 `sampler` 缺失告警——**这条对每个 vLLM 0.29 的 bundle 都成立**
（Qwen3-8B / Zamba2 的 0.29 bundle 同样只有 `lm_head` 没有 `sampler`：
0.29 的 Sampler 不在被 profile 的那段执行路径里），所以它不改变任何跨模型对比的公平性。

**另记一条测量不稳定**：`moe.csv` 里最大的 shot（2048 token）在 r0 上会抖——
同一份配置三次测得 **960 / 1116 / 1206 µs**，而 r4/r128 稳定在 1152 µs；
而 r0 自己 4 专家的 901 µs 比 8 专家的 746 µs 还高，物理上说不通，所以判它是量测
artifact 而不是模型差异（三份 config 的 MoE 段逐字段相同，唯一差别是 attention 的
`compress_ratios`）。对策是合并工具改成**取三份测量的中位数**（原来取第一份，
等于把 r0 的抖动写进 bundle），闸门仍照报 17% 的 spread。

### 步骤 5：接进模拟器 ✅ **已完成（2026-09-22）：四域 bundle 齐、三臂对照跑通**

**判据达成情况**：

```text
# 四类卡各一份 bundle，两个校验器全过
python3 tests/check_profile_bundle.py --hardware {RTX3090,RTX4090,RTX5090,A100} \
        --model casr/P15B --tp 1
python3 tests/check_profile_bundle_consistency.py --model casr/P15B

# 三域正式对照（3P × 3D，与 Qwen3-8B 的对照同拓扑）
CLUSTER_CONFIG=configs/cluster/casr_p15b_three_domain.json \
  bash tests/run_casr_comparison.sh /tmp/p15b-three-domain
```

三臂都跑完，逐层成本全部命中（**唯一剩下的告警是 `sampler`**，见下）：

| 指标（8 请求） | baseline | greedy | LP |
|---|---:|---:|---:|
| latency_mean (ms) | 403.96 | **373.79** | 407.47 |
| latency_p50 (ms) | 456.74 | **431.81** | 466.65 |
| tpot_mean (ms) | 14.35 | **13.05** | 14.48 |
| tpot_p95 (ms) | 15.26 | **13.28** | 15.29 |

（8 个请求、无跨域压力，LP 在这一档本来就该与 baseline 打平；**这一步验的是"真 bundle 能驱动
模拟器"，不是收益**——收益要等长 prompt + 高峰值的正式实验档。）

`sampler` 缺失对所有 vLLM 0.29 的 bundle 都成立（Qwen3-8B / Zamba2 的 0.29 bundle 同样
只有 `lm_head`、没有 `sampler`：0.29 的 `Sampler` 不在被 profile 的执行路径里），
所以它不影响跨模型对比的公平性，但要写进口径。

#### 同负载下的模型对照：KV 小 8.9× 之后，CASR 的符号翻了

两个模型跑**同一拓扑、同一份 80 请求负载**（`casr_hetero_hot_cold.jsonl`，
`static` / `greedy` / `lp` 三臂），脚本 `tests/run_p15b_three_domain_comparison.sh`
与 `tests/run_real_qwen3_8b_three_domain_comparison.sh` 一一对应：

| 模型（KV/token） | 臂 | latency mean | latency p95 | TTFT mean | TPOT mean |
|---|---|---:|---:|---:|---:|
| Qwen3-8B（147,456 B） | static | 2,769 ms | 4,265 ms | 477 ms | 18.0 ms |
| | greedy | 2,849 ms（**+2.9%**） | 4,220 ms | 462 ms | 18.8 ms |
| | lp | 2,897 ms（**+4.6%**） | 4,324 ms | 596 ms | 18.1 ms |
| **P-15B（16,960 B）** | static | 4,004 ms | 6,340 ms | 696 ms | 26.0 ms |
| | greedy | 3,751 ms（**−6.3%**） | 4,950 ms | **76 ms** | 28.9 ms |
| | lp | 3,797 ms（**−5.2%**） | 4,968 ms | **78 ms** | 29.3 ms |

**怎么读**：

* 同一条负载、同一套链路下，**Qwen3-8B 上 CASR 是负收益，P-15B 上转正**——
  正是"KV 缩小一个量级 → 跨域搬运从不可行变成可行"这条主线想要的结果。
  最刺眼的是 TTFT：P-15B 的 static 把 80 个请求全钉在一台 prefill 上（696 ms），
  CASR 把热前缀请求挪到另一台后降到 **76 ms**；Qwen3-8B 因为 KV 太大，
  同样的挪动不划算，TTFT 仍在 460–600 ms。
* **但 P-15B 的静态 TPOT 反而更高**（26.0 vs 18.0 ms）。这不是 bug 而是 MoE 的
  权重流量：每 token 激活只有 1.655B，但 batch 内不同 token 命中不同专家时，
  一步要吃下接近 14.4B × 2 B = 28.75 GB 的权重；Qwen3-8B 稠密只有 16.4 GB。
  也就是说压缩 KV 买到的是"搬运便宜"，代价是"每步权重大"——
  这条必须写进论文的 trade-off，不能只说 KV 小。
* 这一档仍是短 prompt（64 token）、0.4 s 内到齐的突发；**长 prompt + 高峰值
  持续档**才是计划里"收益收敛多少"的正式曲线，上面这组是它的第一个点。

`configs/cluster/casr_p15b_three_domain.json`：与
`casr_real_qwen3_8b_three_domain.json` **同一拓扑**（同样的节点、链路、实例），
只换两样——`model_name: casr/P15B`，以及
`casr.kv_bytes_per_token = 16960`（设计闭式 16.56 KB/token；Qwen3-8B 是 147456）。

服务时间与容量仍是 Qwen3-8B 的占位值：`rescale_service_times` /
`rescale_capacities` 在加载时会用 `profiler/perf/<HW>/casr/P15B/bf16` 覆盖它们，
所以那两个字段只是"锚点"，不是 P-15B 的数。

链路已逐段验证（都在 `astra-sim` cwd 下跑）：

```text
get_config("casr/P15B")          -> 28 层、三种 block type
_load_architecture("p15b")       -> r0 / r4 / r128 三条流水线
plan_layer_sequences(cfg, arch)  -> 28 条逐层流水线，例如
    layer  2 (r4)   : … compressor_csa, kvproj_csa, indexer_csa, rotary_emb, attention, …
    layer  3 (r128) : … compressor_hca, kvproj_hca, indexer_hca, rotary_emb, attention, …
    layer  0 (r0)   : … rotary_emb, attention, …
```

也就是说"每层走自己的压缩模块、查自己的 attention 表"这条链已经通了——
剩下只等四个域的 bundle 落盘。

**一处口径修正（2026-09-22）**：四个 `configs/model/casr/P15B*.json` 一开始**没有
`kv_layout` 块**（只有 `DeepSeek/DSV4-P15B-draft.json` 有），于是 `pd_kv_bytes` 走了
通用分支——按"28 层 × 每层 512 宽 K+V"计价：

```text
修前：2 × 512 × (28 × 10) × 2 = 573,440 B / 10 token   （= 57,344 B/token，全注意力价）
修后：sum(values/token)=8,480 × 2 × 10 = 169,600 B / 10 token   （= 16,960 B/token ✓ 设计值）
```

补上 `kv_layout`（28 层 ratio 列表；三份单类型 config 各 1 层）之后，
**`pd_kv_bytes`（执行侧）与 cluster config 的 `casr.kv_bytes_per_token = 16960`
（计划侧）口径一致了**——这正是 `hw_service` 一直强调的"计划和执行定价同一个引擎"。
如果没发现这条，handoff 代价会被高估 **3.4×**，而 handoff 恰恰是 CASR 决定搬不搬的依据。

完整管线里复核过（同一份 smoke 跑，逐请求 CSV）：

```text
input=10 tok  pd_kv_bytes=169,600   (16,960/token)
input=16 tok  pd_kv_bytes=271,360   (16,960/token)
input=22 tok  pd_kv_bytes=373,120   (16,960/token)
```

——严格按设计值成比例，不再是 57,344/token 的全注意力价。

#### 步骤 5 的第三个缺口：模拟器把 28 层全当第 0 层定价（已修）

配好之后做端到端验证时发现：**按 block type 分派的 attention 表根本没被查**。
把表整体放大 10× 做 A/B：

```text
dense.csv 的 qkv_down ×10     -> Mean TPOT 11.40 → 17.53 ms   （被读了）
attention.csv（r0）×10        -> 29.74 ms                      （被读了）
attention_r4.csv ×10          -> 11.40 ms（纹丝不动）          （没被读）
```

加探针打出每一次 attention emission 的 `layer_num`，真相是**142 次全是 `layer_num=0`**：

`_synthesize_trace` 的 block-copy 优化——**建一层、复制 `num_hidden_layers` 次**。
`config_builder` 只在**集群配置声明了逐 block 放置覆盖**时才关掉它（注释写得很准：
"if weights differ in blocks, we can not copy and paste the same trace for all
layers"），但**模型自身的 `layers_block_type` 没有触发它**。于是 P-15B 的 28 层
全按第 0 层（r0，全 MLA、无压缩模块）定价。

修法：新增 `_can_copy_blocks(ctx, block_mode_on)`——yaml 里有 `layer_types` 就不允许复制，
两条 synthesize 路径都用它。修完的效果：

```text
分派探针：layer 0 → r0，layer 10/12/14 → r4，layer 11/13/15 → r128  ✓
Mean TPOT：11.40 → 13.05 ms   （旧路径低估约 14%）
```

**这条也意味着 Zamba2 那种 hybrid 的既有结果同样受影响**：38 层全按第 0 层（mamba）定价，
6 个 attention 层等于没计费。那批数字需要重跑——`run_hetero_arena.py` 现在会走对路径。

**影响已量化**（同一套条件：Zamba2-1.2B，2 域 `5090,4090`，`--peak-rps 8
--peak-seconds 30`，`rr`/`casr_lp` 两臂；修复前用 `git checkout 4b796f9~1 --
serving/core/trace_generator.py` 跑）：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| `rr` TPOT p50 | 3.4 ms | **4.8 ms（+41%）** |
| `casr_lp` TPOT p50 | 3.4 ms | **4.8 ms（+41%）** |
| `rr` E2E mean | 21,943.7 | 21,982.5 |
| `casr_lp` E2E mean | 22,070.0 | 22,107.5 |

**TPOT 抬高 41%，E2E 几乎不动**——这一档的 TTFT（21.9 s）完全由排队主导，
所以端到端被掩盖了；**凡是拿 Zamba2 的 TPOT（或任何 per-token 指标）做的结论，
都要按这条重跑**。

**按文档原命令复核过（6 个 4090 域、`--peak-rps 8 --prompt-tokens 1250`、五臂）**：

| arm | 修复前 E2E mean (ms) | 修复后 E2E mean (ms) | 变化 |
|---|---:|---:|---:|
| `load` / `cache_aware` | 418,912.9 | 418,958.1 | +0.01% |
| `rr` | 163,767.9 | 163,814.8 | +0.03% |
| `casr_lp` | 65,334.7 | 65,387.2 | +0.08% |
| `casr_full` | 40,355.1 | 40,475.6 | +0.30% |

**端到端结论不变**（`load` 仍把 735 个请求全钉在一台、TTFT 418 s；`casr_full` 仍压到 ~40 s），
因为这一档由 TTFT/排队主导；**变的只有 per-token**：TPOT p50 各臂 **4.9 → 7.1 ms（+45%）**。

#### 步骤 5 闭环：P-15B 的 bundle 能跑完 baseline / greedy / LP 三臂

修完零流量与 block-copy 之后，用 P-15B 的 bundle 跑完整对照入口：

```text
CLUSTER_CONFIG=/tmp/p15b_smoke_cluster.json bash tests/run_casr_comparison.sh <outdir>
  → baseline vs greedy  ✓
  → baseline vs LP      ✓
```

（那次是 1P×1D 的冒烟拓扑，没有可路由的余地，所以三臂数字相同——**能跑完**才是这一步的判据。
正式实验用三域的 `casr_p15b_three_domain.json`，那里 3P×3D 才有路由空间。）

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

### 步骤 1–5 的复验记录（2026-09-22 20:40，当前代码）

模型改过（compressor norm 换类、`o_lora_a` 换成容器），所以三个闸门都在**改动之后**
重跑了一遍：

```text
check_p15b_shapes.py            r0 497,383,616 / r4 503,365,824 / r128 500,415,680  全对
                                parity(fp32, T=256) max|Δlogits| = 9.5e-07 ~ 1.1e-06
check_p15b_catalog_binding.py   每个 canonical 名唯一绑定，无 AMBIGUOUS
check_p15b_boot.py              四卡全过，KV 账目与设计闭式一致
    4090 sm89   1152 / 1024 /  16 B per token
    3090 sm86   1151 / 1023 /  16
    5090 sm120  1152 / 1025 /  16
    A100 sm80   1152 / 1024 /  16
```

### 明确留待下一步的三项（都不阻塞当前实验）

1. **TP=2 的 bundle 没采**。`_linear()` 现在还是 `nn.Linear`（vLLM 并行层只留了
   接口），TP>1 的 profile 会量到"没切分"的形状；而
   `casr_p15b_three_domain.json` 与 Qwen3-8B 的对照拓扑用的都是 `tp_size=1`，
   当前实验不需要它。要做就得先按 §4.1 把 qkv_down / o_lora / MoE / embedding
   换成并行层，再补 tp2 的 dense / per_sequence / attention（attention 一类就是
   一套 4 小时的 sweep）。
2. **A100 的 dense 两次测量差 ~20%**：17:3x 那次与 19:5x 的重采（同一配置、同一
   张卡、都与其他作业同机并发）在大 shot 上整体差两成。bundle 用的是重采那版，
   域内一致性 2.7% 没问题，但**跨域比较在大 batch 档要留意这个量级的不确定度**；
   要收紧就得在整机独占的条件下重采一次做基准。
3. **步骤 6（Triton 稀疏注意力）** 仍是可选性能版，没做。

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
| **一个类名被 catalog 两处匹配**：DedupSink 取平均 → 要么重复计费、要么把 N 个模块平均成 1 个（`o_lora_a` 实测 8 倍低估） | 采集**前**跑 `tests/check_p15b_catalog_binding.py`；合并工具的一致性闸门做二次兜底（这次就是它先报的 17.3%） |
| 12 次 profile 的时间 | 单次与 Qwen3-8B 同量级（`num_hidden_layers=1`，MoE 稍大）；按卡分片并发 |

## 6. 资源与时间估算

- **卡**：现在就有 4090×2（本机）、3090×2（3090a）、5090×2、A100-80G×4（0–3 空）。
- **单次 profile**：与 09-18 那次同量级（30–40 min，`iters 3`、`attention_max_kv 16384`）。
- **总量**：3 类型 × 4 卡 = 12 次；按卡并发、类型串行 → 每卡 3 次。
  ⚠️ **这条估算后来被实测推翻**：attention 阶段是绝对瓶颈（每类 7000+ shot，每次要做
  "gather 缓存 + 窗口 + top-k 联合 softmax"），单域实际 **4–5 小时**，见
  [步骤 4 的正式采集](#步骤-4-的正式采集已启动2026-09-22)。dense / per_sequence 各自
  只要 2–3 分钟，所以后来能用 `--categories dense` 单独重采修正过的行。
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
