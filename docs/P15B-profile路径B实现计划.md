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

### 步骤 5：接进模拟器 ✅ **配置已就位（2026-09-22，等 bundle 落地即可跑）**

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
