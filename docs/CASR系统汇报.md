# CASR 系统汇报：面向跨域异构 P/D 分离的存算协同结构调度

| 项 | 内容 |
|---|---|
| 报告日期 | 2026-09-17 |
| 覆盖范围 | 算法设计、模拟平台、真机实验、混合注意力模型专题 |
| 代码状态 | `LLMServingSim` 仓库 `main`；**本报告数字对应的最后一次代码提交是 `e442909`**（报告本身在其后一次提交落库）；`pytest tests -q` → 243 passed / 2 skipped |
| 一句话 | 把"要不要搬 KV"变成可决策项、并让"加/减一台 Prefill"按全局结构收益来定 |

本报告把分散在十余篇文档里的设计与证据收敛成一篇可独立阅读的系统汇报。
事实与数字的来源逐处标注，**引用时请连同第 1.3 节的边界说明一起使用**。

> ⚠️ **测试对象变更预告（2026-09-18）**：本报告（含小论文稿）里的全部数字都是
> 在 **Qwen3-8B**（KV `147.5 KB/token`）上测的，**这批结果保留为"KV 大档"对照**。
> 接下来的主线实验改用**自研 P-15B**（KV `16.56 KB/token` bf16，小 **8.9×**）：
> 决定、依据与口径见 [DSV4_10-30B跨卡设计.md](DSV4_10-30B跨卡设计.md) §0.1，
> 数据集/基线侧见 [实验数据集与对比基线说明.md](实验数据集与对比基线说明.md) §0。
> **本报告尚未纳入 P-15B 的实验结果**；在小论文稿里引用本报告数字时，
> 需要同时说明它们对应的测试对象是 Qwen3-8B。

> **投稿级细节见 [CASR小论文稿.md](CASR小论文稿.md)**：同一批工作按论文结构
> （摘要 / 相关工作 / 动机测量 / 设计 / 实验 / 讨论 / 参考文献 / 附录）重写，
> 含 18 条已核对 arXiv 编号的引用与 6 张图（`docs/figs/`，由
> `tests/plot_report_figures.py` 生成）。本报告是内部汇报口径，论文稿是投稿口径，
> 两者共用同一批数字，改数时需同时更新。

---

## 摘要

P/D 分离把一次请求切成两段，也把一个被经典负载均衡忽略的变量变成主要成本：
**KV 要从 Prefill 搬到 Decode**。在这套真实 fabric 上，搬一次 1000-token 的 KV
要 **0.6 s（同主机）～1.4 s（跨域）～8.2 s（A100 慢链路）**，而 Decode 自己把
这段 prompt 重算一遍只要 **约 93 ms**——差 6～88 倍。经典路由（least-loaded /
前缀感知）从不问"要不要搬"，只问"搬到哪台"，于是**无论怎么选实例都要付这笔钱**。

本工作（CASR）在同一套 vLLM + NIXL 基座上增加一个**慢时间尺度**的结构调度层：

1. **快层（每请求）**：在计划给出的亲和域内选 P/D pair，并逐请求决定
   **搬 KV 还是让 Decode 本地重算**；
2. **慢层（1 s 控制周期）**：解一个带**链路字节预算**的流量分配 LP，
   生成带 TTL 的亲和计划；再用反事实评估决定 **±1 台 Prefill** 的启停。

主结果（真机，5 台机器 4P×3D，每格多轮均值）：`casr_lp` 相对 `load` /
`cache_aware` 在六个档位上 **mean −62.9% ~ −83.6%、p95 −81.6% ~ −93.5%**，
TTFT ≤500 ms 达标率从 16–70% 提到 94–100%；换成压缩 KV（MLA
31.1 KB/token）后优势收窄到 **−25.3%**，这条单调关系本身就是"存算协同值多少"
的刻度。

模拟侧本轮补齐了**真混合注意力模型**的支持（Zamba2-1.2B，4090 与 5090 两份
profile bundle），并在同构/异构/长 prompt/热点前缀四类环境上做了六臂对照，
得到的结论与真机一致的方向是：**瓶颈是 KV 出口，而不是算力**。

---

## 1. 研究问题与定位

### 1.1 问题

部署形态：多个域（机房/机型），每个域一台或多台机器，每台机器上跑一对或多对
**Prefill 实例与 Decode 实例**，中间用 RDMA/NIXL 传 KV。给定请求流，要回答三个
层次的问题：

| # | 问题 | 时间尺度 | 谁在决策 |
|---|---|---|---|
| Q1 | 这个请求走哪对 P/D？ | 每请求 | 快层路由 |
| Q2 | 这条请求的 KV 是搬过去，还是让 Decode 自己算？ | 每请求 | 存算协同决策 |
| Q3 | 当前这套 P-D 结构还值得维持吗？要不要加/减一台 P？ | 秒级 | 慢层结构调度 |

已有系统大多只回答 Q1（请求级路由），少数回答"P/D 配比"（按利用率扩缩容），
而 Q2 与 Q3 的联合——**在 Prefix 状态改变 P 的实际供给之后，结构重构是否值得**
——是本工作的位置。

### 1.2 与近期工作的分工

按**决策时间尺度**分四片（详细映射见
[实验数据集与对比基线说明.md](实验数据集与对比基线说明.md) 第 3 节与
[vLLM_LMCache存算协同调度方案与SOTA对比.md](vLLM_LMCache存算协同调度方案与SOTA对比.md) 第 5、9 节）：

| 片 | 解决什么 | 代表工作 | 与本工作的关系 |
|---|---|---|---|
| 请求级路由 | 请求发给哪个实例 | SGLang Router `cache_aware`、vLLM production-stack prefix-aware、Mooncake KV-aware | **直接对照**（`cache_aware` / `load` / `kv_aware`） |
| P/D 解耦与配比 | P/D 数量与放置 | DistServe、Splitwise、DOPD 类 autoscaling | 复现其"固定配比 + 静态放置"设定作为基线 |
| KV 数据面 | KV 存哪、怎么传 | LMCache、Mooncake Transfer Engine、NIXL | **被复用的传输层**，不是比较对象 |
| 生产编排平台 | worker 生命周期 | NVIDIA Dynamo、llm-d | 平台执行 `scale/drain/start`，CASR 是可插拔慢层策略 |

需要一句话讲清边界：**我们不比"KV 传得更快"，我们比"这一次传不传、这一行该不该加"。**

### 1.3 本报告的边界（引用前必读）

* 真机主表（第 5.1 节）在 `LOCAL_PREFILL_BASELINES=never` 的旧默认下测得：
  基线的请求 **100% 走搬运**，而 CASR 系列可以选择本地重算。
  同一 trace 同一 fabric 上这条差异达 `1394.1 → 345.9 ms`。
  **因此"相对 load"的百分比里包含了一条只对基线关闭的路径**，
  默认值已改为 `auto`，主表待在新默认下重跑。引用旧表必须带这句。
* 模拟器竞技场（第 5.2–5.4 节）中 `casr.local_prefill = never`，**所有 arm 一律
  走 P/D 搬运**，因此那几张表比较的是**放置与结构**，不含 Q2，两边口径不可混用。
* 结构弹性在默认拓扑（Prefill 未饱和、允许本地重算）下**没有收益**；
  在"A100 节点被机主回收"之后，5P×4D 档与 MLA/Zamba2 档**无法补轮**。
* 所有对比都是**同一拓扑、同一 trace、同一批容器**、每个 policy 跑前重建实例
  并过输出正确性门禁；结论仅在该条件下成立。

---

## 2. 系统设计

### 2.1 架构

```text
                       慢层（1 s 控制周期）
  ┌───────────────────────────────────────────────────────────────┐
  │ PrefixProfiler.snapshot(t)  →  PrefillLifecycle.update(t)      │
  │        →  流量 LP（含链路字节预算）  →  AffinityPlan(+TTL)      │
  │        →  反事实评估：+P / -P / warm / cold                     │
  └───────────────┬───────────────────────────────┬───────────────┘
                  │ affinity                       │ scale/drain
                  ▼                                ▼
  请求 → 路由（快层）→ Prefill 实例 → [Q2: 搬 KV / 本地重算] → Decode 实例
                                  └──── NIXL / LMCache ────┘
```

基座分工：**vLLM** 提供 continuous batching、Prefix Caching、KV Connector；
**LMCache/NIXL** 提供外部 KV 与传输；**CASR 控制器**决定上面那三层决策。

### 2.2 符号与状态

| 符号 | 含义 |
|---|---|
| `C` / `P` / `D` | prefix class 集合 / 可接收请求的 Prefill / Decode worker |
| `λ_c`,`h_c`,`r_c` | class `c` 的 EWMA 到达率、命中 token、请求 token |
| `f_{c,p,d}` | class `c` 经 Prefill `p` 交给 Decode `d` 的流量 |
| `K_p`,`K_d`,`B_l` | Prefill / Decode 容量、共享链路容量 |
| `work_c = max(0.05, 1 − min(0.95, h_c/r_c))` | 未命中工作比例 |

`prefix_id` 由"最多 512 token 对齐后取 BLAKE2 摘要"得到，**快照只存摘要**，
因此统计复用不会把请求内容写进实验产物。

### 2.3 固定结构下的流量分配（Q1）

```
min  Σ distance(p,d)·f_{c,p,d} + penalty·(Σ s_p + Σ s_d + Σ s_l)
s.t. Σ_{p,d} f_{c,p,d} = λ_c
     Σ_{c,d} f_{c,p,d}·work_c ≤ K_p + s_p
     Σ_{c,p} f_{c,p,d}        ≤ K_d + s_d
     Σ_{(p,d)∈l} f_{c,p,d}    ≤ B_l + s_l
     f, s ≥ 0
```

三类松弛变量让 LP 在过载时**先牺牲最不痛的那条约束**，并把 overflow 记进诊断
（OR-Tools 缺失时自动回退 greedy）。

**两个关键的容量口径**，都是这轮实验里被数据逼出来的：

1. **Prefix 状态改变 Prefill 的有效容量**：命中 95% 的 worker 与冷 worker
   能接的流量不是同一个数（`work_c` 折进去）。
2. **生产者出口是硬约束**：实测算术上 Prefill 出口只有 **0.26 GB/s**，
   1250-token 请求要推 **157–184 MB** ⇒ **每台 Prefill 约 1.4–1.7 req/s**，
   而同一张卡的**算力**容量是 **39 req/s**。两者差 24 倍——把容量按算力算，
   LP 会认为还有余量，于是 `+P` 的反事实收益显示不出来（实测：+P 臂 14878 ms
   vs 用出口口径容量后的 1400 ms）。

### 2.4 从 flow 到在线路由

`f_{c,p,d}` 转成两级权重：

```
P_weight(c,p) = Σ_d f_{c,p,d} / Σ_{p,d} f_{c,p,d}
D_weight(p,c,d) = f_{c,p,d} / Σ_d f_{c,p,d}
```

计划带单调 `version` 与 `expires_at_ns`，过期或 class 不匹配即回退普通路由，
避免旧计划无限期影响新流量。Prefill 侧用 **deterministic deficit routing**
（`desired − observed` 最大的 worker 优先），比随机加权更可复现。

### 2.5 存算协同决策（Q2）

每个请求在"搬 KV"与"Decode 本地重算"之间取小：

```
transfer_ms = same_node ? 0.585·k : (48 + 1.351·k)      # k = prompt tokens / 1000
local_ms    = 0.093·k + queue_externality(D)
选择 local_ms < transfer_ms
```

其中 `queue_externality` 是"这个请求跑到 D 上重算，会给 D 的队列再加多少延迟"，
防止在 D 已经不空的时候继续塞。

### 2.6 结构层（Q3）

控制 tick 的处理顺序是**先生命周期、后求解**，保证新计划只引用当前真的能接活的
worker；候选动作是 `±1 台 P`（含 warm/cold 两种形成方式），用时间窗把
"多一台 P 的收益"与"这次重构的成本"（容器启动、KV 复制）折算到同一单位再比。
工程上有三条安全规则：`|ΔP| ≤ 1`、dwell time、最小收益阈值。

---

## 3. 实验平台

### 3.1 真机集群

| 域 | 机器 | GPU | 备注 |
|---|---|---|---|
| 5090 | `10.212.70.196` | 4×RTX 5090 32 GiB | 主力 Prefill |
| 3090a/b | `10.212.67.68` / `10.212.70.38` | 2× / 1× RTX 3090 | |
| 4090 | `10.212.67.167` | 4×RTX 4090 | 本项目控制面所在 |
| A100 | `10.66.0.15:2222` | 8×A100 80 GiB（共享机） | **2026-09-14 被机主回收**，数据有效、不能补轮 |

链路实测：同主机 P/D 交接 **585 ms/1000 token**，10.212 跨域 **1351 ms**，
A100 慢链路 **8223 ms**；生产者出口 **0.26 GB/s（239–314 MB/s）**。

### 3.2 模拟器（LLMServingSim）

模拟器不是"另写一个模型"，而是**用同一套 profiler 产物定价**：per-instance 的
服务时间、容量、router 容量、token 速率都来自 `profiler/perf/<hw>/<model>/<variant>`
里的 bundle，ASTRA-Sim 负责网络，因此控制面"计划"和时序"执行"用同一批常数。

本轮为**真混合注意力模型**补了三块（细节见
[混合注意力模型profile.md](混合注意力模型profile.md) 第 5 节）：

1. **逐层 block type**：`profiler/models/zamba2.yaml` 新增 `layer_types`，
   trace 按模型 config 的 `layers_block_type[i]` 选流水线（`mamba` 层只跑
   `[mamba_norm, mamba_mixer]`，`hybrid` 层跑完整共享注意力 + Mamba）。
   在此之前，38 层会被全部当成注意力块——注意力被算 38 次而不是 6 次。
2. **交错 KV 布局**：`kv_geometry.full_layer_indices` 支持 KV 层是每隔 6 层一个，
   而不是"前 N 层"。
3. **尺寸与权重**：认识 `mamba_norm` / `attention_norm` / `mamba_mixer` /
   `shared_linear`，并接受 `attention_head_dim` 与 `attention_hidden_size`
   （Zamba2 的注意力通路 4096 宽而 hidden 只有 2048，弄错 qkv 权重会差一倍）。
   控制面定价函数 `step_cost_ns` 也走同一份逐层流水线，否则"计划按 38 个注意力块
   估容量、时序按 6 个跑"，同一台机器两个数。

### 3.3 Profiler 与多卡并行采集

一个 shot 的固定开销是 **~3.5 s 的 `torch.profiler` 记账，与形状无关**
（MNBT=32/MSQ=2 的 20 个 dense shot 也是 3.5 s/个）。因此一个 4166 shot 的
attention 网格就是 4 小时，只能靠"少发"或"并行"缩短。

本轮新增 `--shard i/N`：保留组合网格中位置 `≡ i (mod N)` 的 shot，在多张卡上
各跑一片再合并。合并脚本**断言分片两两不交且覆盖完整**（重叠或漏采都会产出
"能加载但悄悄错"的 bundle）。实测：Zamba2-1.2B 在 5090 节点两张空闲卡上，
每片 dense 76 + per_sequence 20 + attention 431 shot，**约 37 分钟**完成。

---

## 4. 实验方法

### 4.1 数据集

| 档 | 来源 | 前缀复用 | 用途 |
|---|---|---|---|
| Dolly-15k | `databricks/databricks-dolly-15k` | **几乎为零** | 把"容量/异构感知"的贡献从"前缀亲和"里剥出来 |
| ShareGPT GPT-4 filtered | `shibing624/sharegpt_gpt4`（多轮切分） | **强**（本轮输入=前几轮上下文） | 前缀缓存/亲和路由的主战场 |
| CNN/DailyMail | `abisee/cnn_dailymail` 3.0.0 | 零 | 长上下文；单请求 KV 约 184 MB |
| 合成长 prompt | `make_phased_trace.py --pack N`（拼接 N 篇） | 零 | 把 KV 推到 402 MB/请求 |
| 热点前缀 | `make_hot_prefix_trace.py`（2×1024 共享头） | 强 | **唯一能把"缓存感知路由"与"最空实例"区分开的负载** |

一条被实测确认的方法论：**零复用数据集上 `cache_aware` 与 `load` 逐字节相同**
（同一个 735 请求集合、同一落点、同一延迟）。因此不能用零复用数据集去评价
缓存感知路由——这不是"缓存没用"，而是这份负载里缓存不可能有用。

### 4.2 基线

| arm | 对应 | 说明 |
|---|---|---|
| `rr` | 经典轮询 | 隔离"完全不管缓存与队列" |
| `load` | vLLM production-stack / Llumnix 的队列语义 | `(inflight+1)/capacity` |
| `cache_aware` | **SGLang Router** `cache_aware` | 最长前缀优先，命中实例忙则溢出到最空 |
| `kv_aware`（本轮新增） | Mooncake/LMCache 式"按 KV 定价"的 **P 侧形态** | 打分用 `max(算力等待, 出口排队+本次串行化)` |
| `casr_lp` / `casr_full` | 本工作 | 静态池 / 叠加结构弹性 |

`kv_aware` 的存在意义是回答一个具体的反驳：**"你的基线只是没调好"**。
它属于同一族路由，只把打分换成绑定资源，就能拿回 2.24×（第 5.3 节）。**归因修正（2026-09-17，评审 R9）**：下面这条 2.24× 优势**不是**打分函数带来的——同一打分的 `load_rr`（只把决胜改成轮询）得到 163.7 s，与 `kv_aware` 的 175 s 等效。因此第一段应记为 tie-break 修复，第二段（字节预算规划，2.51×）才是本工作的主张。详见 `CASR小论文稿.md` §6.9。

### 4.3 环境（四个正交旋钮）

| 旋钮 | 参数 | 检验 |
|---|---|---|
| 硬件域 | `--domains 5090,5090,4090,4090,4090,4090` | 算力/显存异构 |
| 跨域链路 | `--cross-gbps` / `--cross-rtt-ms` | 广域下的放置 |
| 池深度 | `--min-active` + `inactive_instances` | 弹性头寸，且**基线不得偷跑** |
| KV 出口预算 | `--kv-egress-gbps` | 通道主导程度 |

### 4.4 SLO 口径与正确性

* **SLO**：TTFT 与 TPOT 两段分别计时；真机按 `slo_ttft_ms` / `slo_tpot_ms`
  逐请求判定，模拟器复用同一口径。
* **正确性门禁**：长文档在首尾埋随机校验码比对；混合/压缩 KV 模型额外用
  "本地生成 vs P/D 生成"一致性判据。跑测前重建实例、预热全部 P/D 对。
* **统计口径**：逐请求延迟取各轮 `summary-<policy>.json`（不用 router 的
  `metrics-*.jsonl`，后者含门禁请求且跨目录去重会漏轮次）。

---

## 5. 结果

### 5.1 真机主结果（4P×3D，多轮均值，ms）

> **口径**：见第 1.3 节第 1 条——基线被禁止本地重算。引用需带此说明。

| 档位 | `load` | `cache_aware`（SOTA） | `casr_lp` | mean Δ | p95 Δ |
|---|---:|---:|---:|---:|---:|
| CNN-short 14 req/s | 880.2 | 919.5 | **281.8** | **−68.0%** | −81.6% |
| Dolly-15k 零复用 | 770.6 | 802.4 | **286.0** | **−62.9%** | −83.3% |
| ShareGPT deep 强复用 | 2725.6 | 3522.8 | **579.2** | **−78.8%** | −93.4% |
| CNN-short 56 req/s | 2830.1 | 2784.9 | **568.0** | **−79.9%** | −93.5% |
| CNN-XL 3800 token | 20274.8 | 15982.1 | **3320.6** | **−83.6%** | −92.6% |
| 五域异构 5P×4D | 1214.0 | 1213.1 | **291.5** | **−76.0%** | −84.5% |
| 压缩 KV（MLA 31.1 KB/token） | 1101.4 | 1067.5 | **822.3** | **−25.3%** | −27.7% |

三个附加事实：

* **XL 档基线会丢请求**：3 轮累计完成率 `load 346/360`、`cache_aware 246/360`，
  而 `casr_lp 360/360`。
* **KV 越小，相对优势越小**：同一模型 bf16→fp8（147.5→73.7 KB/token）后基线快
  10–40%，`casr_lp` 只动 ±2%；MLA 档优势降到 −25.3%。这正是"存算协同值多少"
  的刻度尺。
* **归因**：把决策规则也给基线，`load` 从 886.3 降到 344.1（−61%）——
  **"有没有这个决策"是主要贡献**，在此之上路由/代价模型再贡献 −14.5%
  并显著改善尾部（p95 506.5 → 335.1）。

### 5.2 模拟：真混合模型（同构 6×RTX4090，1250-token，8 rps）

单位秒。原始 `summary.json` 归档在 [`docs/arena-summaries/`](arena-summaries/)。

| arm | Qwen3-8B（全注意力） | Zamba2（真混合） |
|---|---:|---:|
| load | 210.5 | 418.9（p95 803.9） |
| cache_aware | 210.5 | 418.9 |
| kv_aware | 227.3 | **174.7** |
| rr | 200.6 | 163.7 |
| casr_lp | 85.0 | 65.3 |
| casr_full | 54.9 | **43.3**（p95 89.1） |

### 5.3 模拟：硬件异构（5090×2 + 4090×4，1250-token，8 rps）

| arm | E2E 均值 | p95 | TTFT p50 | span | Prefill 落点 |
|---|---:|---:|---:|---:|---|
| load | 403.0 | 772.0 | 402.7 | 920.4 | 735 → 实例 0 |
| cache_aware | 403.0 | 772.0 | 402.7 | 920.4 | 同上 |
| kv_aware | 179.7 | 363.6 | 176.7 | 488.5 | 372 / 363 |
| rr | 163.7 | 312.5 | 163.2 | 436.8 | 368 / 367 |
| casr_lp | 65.3 | 124.4 | 65.2 | 239.7 | 370 / 365 |
| casr_full | **45.2** | **93.8** | 45.4 | 205.5 | 314/311/91/19 |

TPOT p50 从 3.3 ms 降到 **2.2 ms**：CASR 的 pair 代价把 Decode 更多放到 5090 域，
基线对此完全不敏感。

### 5.4 模拟：长 prompt 与热点前缀

**长 prompt（`--pack 3`，3750 token，KV 402 MB/请求）**

| arm | E2E 均值 | p95 | span |
|---|---:|---:|---:|
| load | 293.5 | 570.6 | 691.8 |
| rr | 246.1 | 472.6 | 562.2 |
| casr_lp | 119.2 | 229.5 | 304.4 |
| casr_full | **93.6** | **194.2** | 266.8 |

**热点前缀（2×1024 共享头）**

| arm | 局域网（0.11 GB/s / 48 ms） | 广域（0.05 GB/s / 80 ms） |
|---|---:|---:|
| load | 104.3 s | 410.2 s |
| cache_aware | 104.3 s | 410.2 s |
| rr | 2.74 s | 65.4 s |
| **casr_lp / casr_full** | **0.413 s** | **0.413 s（逐位不变）** |

三个 arm 的 `npu_hit_tokens` 都 ≈1021（命中率 81.7%）、`pd_kv_bytes` 都是
157 MB/请求：**缓存机制在所有 arm 下都在工作，差别只在放置**。跨域链路变慢
4–24× 时 CASR 一个数位都不变，因为它把跨域字节代价记进了边权。

### 5.5 机制归因：把差距拆成三段

以硬件异构 1250-token 档为例：

```
403.0 s  load / cache_aware      ← 算力型打分：算力越快越退化，实例 id 决定一切
179.7 s  kv_aware                ← 只把打分换成"绑定资源"          → 2.24×
163.7 s  load_rr                  ← 同一打分，只把 tie-break 改成轮询 → 与上一行等效
 65.3 s  casr_lp                 ← 再按字节预算做全局规划            → 2.75×
 45.2 s  casr_full               ← 再加结构弹性                      → 1.44×
```

第一段说明**原来的 SOTA 路由不是被更强的规划器打败的，是被一个看错资源的打分
函数打败的**；第二、三段才是本工作的主张。而在 Qwen3-8B 上 `kv_aware` 反而比
`load` 差 8%（227.3 vs 210.5 s）——因为它的 Prefill 要 185 ms，`inflight` 有真实
起伏，算力型打分**碰巧**还分得动。**指标退化越严重、改回来才越有用**，
而"算力快 + KV 重"正是混合注意力模型的方向。

---

## 6. 关键发现

1. **"搬不搬 KV"比"放在哪"更值钱。** 在实测 fabric 上搬运比重算贵 6–88 倍，
   基线 100% 走搬运、`casr_lp` 在这些档位 100% 选本地；把决策规则交给基线，
   基线自己也快了 61%。
2. **Prefill 的真实容量是 KV 出口，不是算力。** 0.26 GB/s ⇒ 1.4–1.7 req/s/台
   vs 算力 39 req/s，两者差 24 倍。按算力估容量会让 `+P` 的反事实收益消失。
3. **算力型负载均衡在"算力快 + KV 重"的模型上结构性失效。** Zamba2 的 Prefill
   只要 31 ms，`inflight` 立刻归零，打分对每台实例都是 ~0，实例 id 成了唯一
   区分——735 个请求全钉在一台上（TTFT 419 s）。**修正：这主要是确定性 tie-break
   造成的**（同分数的 `load_rr` 即回到 163.7 s 且落点 368/367），不是打分函数本身的缺陷。
4. **真混合模型省的 KV 远没有"层数比例"暗示的多。** Zamba2 只有 6/38 层带 KV，
   但那 6 层是 MHA（每 token 16 KB），Qwen3-8B 是 GQA（4 KB）——
   **净省只有 0.85×**。合成档的 2.5× 来自"窗口层几乎不占 KV"这个不成立的假设。
5. **零复用数据集上 `cache_aware` 与 `load` 逐字节相同。** 评价缓存感知路由
   必须换负载（热点前缀/多轮对话），否则会得出错误的"缓存没用"结论。
6. **结构感知的价值随环境变差而放大。** 跨域链路放慢 4–24× 时，`rr` 慢 23.9×、
   `load` 慢 3.9×，而 CASR 逐位不变——因为它按字节预算把 P/D 配成同域对。

---

## 7. 工程缺陷与修复记录

这一节保留下来，是因为**每一条都曾让某个结论失效**，且都写进了回归测试。

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 模拟 TTFT 1898 ms（真机 50 ms） | KV 字节被写在每个 `qkv_proj` 行上，转换器逐行生成 SEND/RECV，一次交接被收 36 次 RTT | 把整个 batch 的 KV 聚合到一行，延迟只收一次（`tests/test_trace_kv_handoff.py`） |
| 2 | 长 prompt 下模拟器**不再前进**：`Running 0 / Waiting 104 / 0.0 tokens/s`，时钟空转到 3973 s | P/D 交接缓冲区按**批次**计费、按**请求**释放；3750-token prompt 分 2 chunk 构建 ⇒ 每请求漏 375 MB，32 GB 缓冲在约 85 个请求后填满 | 改为**准入时按请求计一次**、按计费实例释放；新增 `SIM_ADMISSION_DEBUG` 让闸门自报资源名（`tests/test_pd_staging_budget.py`） |
| 3 | 基线拿到比算法更多的容器 | `--min-active` 只约束 CASR 的生命周期，`load`/`rr` 不理会它，可在全部 6 台 Prefill 上路由 | 生成的集群配置写 `inactive_instances`，spare 对所有策略不可用、控制器仍可拉起（热点前缀档 rr 从 174 ms 变 2.74 s，**16×**） |
| 4 | `+P` 的反事实收益显示不出来 | 容量按算力计（9.3 rps）而非出口（1.4 rps），多一台 P 看起来没用 | 控制器用**出口口径容量**覆盖 LP 的 `K_p`（+P 臂 14878 ms → 1400 ms） |
| 5 | 真机 `cache_aware` 与模拟器语义不一致 | 模拟器取"第一个空闲命中者"，真机按"该请求将造成的负载"排序 | 对齐 tie-break 语义（`tests/test_router_cache_aware.py`） |
| 6 | 高负载档 LP 变体异常 | 流式 Decode 记账 + LP 的负载项缺陷 | 见 [实验结果汇总.md](实验结果汇总.md) §5.5.1 |

---

## 8. 结论与后续工作

### 8.1 可以做的主张

* 在**真实跨主机 P/D 集群**上，把"要不要搬 KV"显式建模并逐请求决策，相对
  least-loaded 与 SOTA 前缀感知路由取得 mean −62.9%~−83.6%、p95 −81.6%~−93.5%，
  并在四个档位做到 100% 请求满足 `TTFT ≤ 500 ms`（基线 16–70%）。
* 在**模拟器**上，把 tie-break 换掉能拿回 2.56×（419 → 164 s），**再叠加按字节
  预算的全局规划与结构弹性才拿满**（2.51× 与 1.51×）。修正后的拆解回答了
  "SOTA 路由差在哪"：**不在打分函数，而在规划与结构**。
* **KV 表示是被解释变量**：KV 越小（fp8、MLA），我们的相对优势越小且单调
  （−68% → −25.3%），给出"存算协同值多少"的刻度。

### 8.2 不能做的主张（务必避免）

* 不能说"首次用 vLLM/LMCache 做 P/D 分离"——那是基座能力。
* 不能说"联合考虑缓存/网络/异构/扩缩容"——过度重叠。
* 不能把 `cache hit rate` 当主胜利指标——那会被读成缓存工程。
* 引用真机主表必须带 `LOCAL_PREFILL_BASELINES=never` 的口径警告；
  结构弹性只能表述为"有热备 + 池子饱和"场景下的收益。

### 8.3 后续工作（按优先级）

1. **在新默认 `local_prefill=auto` 下重跑真机主表**，让基线与算法共享同一组
   选项，消掉第 1.3 节的口径警告。
2. **补齐两格对照**：3750-token 档的 `kv_aware`、1250-token 档的 Qwen 六臂表。
3. **补 3090 / A100 的混合模型 bundle**（继续用 `--shard` 多卡并行），
   让真混合模型能进更多异构维度。
4. **把"绑定资源"打分回写到真机基线**并复测，决定它是否作为新的 SOTA 基线，
   同时给审稿人一个"只修指标能拿多少"的上界。
5. **固定 SOTA 的论文版本与仓库 commit**（当前对照表只有系统名，没有版本号）。

---

## 附录 A 复现入口

```bash
cd LLMServingSim
# 1) 多卡并行 profile
docker run ... -m profiler profile Zyphra/Zamba2-1.2B --hardware RTX5090 \
    --tp 1 --dtype bfloat16 --out-root /out/card0 --shard 0/2 --skip-skew ...
python3 tests/merge_profile_shards.py --out-root profiler/perf \
    --hardware RTX5090 --model Zyphra/Zamba2-1.2B --variant bf16 --tp 1 /out/card0 /out/card1
python3 tests/check_profile_bundle.py --hardware RTX5090 --model Zyphra/Zamba2-1.2B --tp 1

# 2) 六臂竞技场（同构 / 异构）
python3 tests/run_hetero_arena.py --out /tmp/run \
    --domains 5090,5090,4090,4090,4090,4090 --model Zyphra/Zamba2-1.2B \
    --peak-rps 8 --prompt-tokens 1250 \
    --arms load,cache_aware,kv_aware,rr,casr_lp,casr_full

# 3) 长 prompt / 热点前缀 / 广域
python3 tests/run_hetero_arena.py --out /tmp/run3 --pack 3 --peak-seconds 45 ...
python3 tests/run_hetero_arena.py --out /tmp/runhp \
    --trace workloads/cnndm-hotprefix-arena-8rps.jsonl ...
python3 tests/run_hetero_arena.py --out /tmp/runwan --cross-gbps 0.05 --cross-rtt-ms 80 ...

# 4) 回归
python3 -m pytest tests -q          # 243 passed, 2 skipped
SIM_ELASTICITY_E2E=1 python3 -m pytest tests/test_sim_elasticity_end_to_end.py -q
```

## 附录 B 文档地图

| 主题 | 文档 |
|---|---|
| 算法形式化 | [CASR核心算法设计.md](CASR核心算法设计.md)、[CASR调度算法实施设计.md](CASR调度算法实施设计.md) |
| 真机结果与口径 | [实验结果汇总.md](实验结果汇总.md)、[CASR测试报告.md](CASR测试报告.md) |
| 模拟器对齐与能力差距 | [模拟器与真机一致性核查.md](模拟器与真机一致性核查.md)、[对齐矩阵.md](对齐矩阵.md) |
| 大规模异构环境 | [大规模异构模拟环境.md](大规模异构模拟环境.md) |
| 数据集与基线 | [实验数据集与对比基线说明.md](实验数据集与对比基线说明.md) |
| 混合注意力模型专题 | [混合注意力模型profile.md](混合注意力模型profile.md)、[混合注意力模型的调度实验.md](混合注意力模型的调度实验.md)、[混合注意力模型工作总结.md](混合注意力模型工作总结.md) |
| SOTA 边界与定位 | [vLLM_LMCache存算协同调度方案与SOTA对比.md](vLLM_LMCache存算协同调度方案与SOTA对比.md) |
