# 近一年异构 P/D 分离调度的路线，与模拟器里真正跑得起来的基线

本文回答两个问题：**近期（2024-09 起的 [year]）异构 P/D 分离调度研究分几条路线**，
以及**其中哪些机制已经在 `LLMServingSim` 里实现成了可跑的对照臂**、还差什么。

> 版本会漂移：本文按 2026-09-24 的仓库状态写，引用的是**机制**而非具体数字。
> 所有"已实现"都指仓库里能跑的 flag / arm，见文末的复现命令。

## 1 路线划分与覆盖情况

先把"近一年"具体化到**系统级**（时间取公开时间，写作时需再核对 venue/版本——
本项目比较的是**机制**，不是复现某个 commit）：

| 系统（公开时间） | 它解决的调度问题 | 我们采样的机制 | 我们的对照臂 |
|---|---|---|---|
| DistServe（2024） | P/D 解耦 + 按 SLO/goodput 定 P:D 配比 | SLO 约束下的容量分配 | **`distserve`**：`tests/pd_ratio_search.py` 按 goodput 搜配比、再按机器数取最小，写出的池交给 `load` 跑（`distserve_lp` 用我们的 LP 跑同一个池） |
| Splitwise（2024） | 机器级 prompt/decode 分池 | 分池 + 阶段迁移 | 同上（配比搜索即"分池"决策；池内路由仍是最简 `load`） |
| SGLang / RadixAttention（2024–2025） | 前缀复用 + cache-aware 路由 | 最长前缀匹配优先、忙则溢出 | **`cache_aware`**（与真机 `_pick_cache_aware` 同语义） |
| Mooncake / KVCache-centric（2025） | 以 KV 为中心的数据面 + transfer engine | 按 KV 字节与链路占用定价 producer | **`kv_aware`** |
| NetKV 类 KV-aware decode 选择 | P 固定后按 queue/网络/KV 选 D | 网络+队列感知的 D 选择 | `_decode_cost_select`（CASR 内部；非独立臂） |
| LMCache / NIXL（2024–2025） | KV 存取与传输 | 逐请求"搬 vs 本地重算" | 所有臂共享的 `local_prefill: auto` |
| DOPD 类动态 P/D autoscaling | 按需求调整 P/D 数量 | 需求/利用率阈值扩容 | **`dopd`**（`serving/casr/autoscalers.py::ThresholdScaler`，利用率/队列阈值 + 迟滞）；`casr_elastic` 是反事实收益版 |
| Llumnix（2024） | 实例内请求迁移 | 迁移在途请求 | **`llumnix`**（`serving/core/migration.py`）：队列重平衡 + KV 搬运计费；**未建模**实例内 block 级搬移 |
| ServerlessLLM（2024） | 快速 checkpoint 加载/启动 | 缩短实例启动 | **`serverless_elastic`**：`serving/core/boot_model.py` 用"引擎初始化 + 权重/存储带宽"推启动，标定到 6.25 的实测重启 |
| MemServe（2024）/ TetriInfer（2024）/ Sarathi-Serve（2024） | 前缀感知调度、chunked prefill | chunked prefill、前缀亲和 | chunked prefill 在时间线里（`--max-num-batched-tokens`）；前缀亲和见 `cache_aware` |
| PrfaaS 类跨 DC prefill 卸载（2025） | 跨数据中心 prefill + 带宽/缓存感知 | 跨域阈值与 P/D 比例调整 | **`prfaas` / `prfaas_tight`**：本地域优先 + `--prfaas-max-offload-ms` 硬预算（阈值本身可扫） |
| NVIDIA Dynamo（2025） | worker/router/传输组件化平台 | 可插拔编排底座 | 只作底座（`PrefillLifecycle`/`ReconfigExecutor`） |

| # | 路线（代表系统） | 该路线的核心机制 | 我们的对照臂 | 覆盖度 |
|---|---|---|---|---|
| R1 | **P/D 解耦与配比**（DistServe、Splitwise 一类的分离式服务） | 把 prefill 与 decode 放到不同实例，按 TTFT/TPOT 预算与 goodput 定 P:D 比例 | **`distserve`**（`tests/pd_ratio_search.py`：按 goodput 搜配比 → 同 goodput 取最少机器 → 写配置）+ `distserve_lp`；`static`/`casr_plan3`/`casr_static` 作固定池参照 | **已实现**，但**受拓扑限制**：域式集群每个节点固定 1P+1D（`intra_node_link_bw` 要求节点布局一致），能表达的只有 `n:n`，搜索实际决定"留哪几台节点"；非均匀配比要单节点拓扑（arena 上验过 1:2 vs 2:2） |
| R2 | **前缀感知路由**（SGLang / RadixAttention 一类 cache-aware router） | 最长前缀匹配优先，命中者优先，忙则溢出到最闲 | `cache_aware`（`--request-routing-policy CACHE_AWARE`）：与真机 `_pick_cache_aware` 同语义（含 `CACHE_AWARE_OVERFLOW=0.75` 溢出阈值） | **已实现**（语义对齐真机路由器） |
| R3 | **KV/网络感知路由**（Mooncake / NIXL / LMCache 一类以 KV 为中心的数据面；NetKV 一类网络感知 decode 选择） | 按 KV 传输代价、链路占用、队列选边；把 KV I/O 当作一等成本 | `kv_aware`（`KV_AWARE`）：按 producer 出口排队 + 序列化时间与算力等待取 max | **已实现** |
| R4 | **传输 vs 重算的存算协同**（LMCache/NIXL 的"搬还是不算"决策，真机 `disagg_router.kv_exchange_decision`） | 逐请求比较"搬 KV"与"本地重算 prompt"，两边都带各自的排队外部性 | 所有臂共享：`local_prefill: auto` + `_link_derived_terms`；CASR 臂另加计划 | **已实现**（本次修复后所有臂同价，见 §3） |
| R5 | **动态 P/D 扩缩容**（DOPD 一类按需求/利用率调整实例数） | 需求/利用率超阈值就加实例，低了就删 | **`dopd`**（利用率/队列阈值 + 迟滞 + hold 窗口，`structural.rule="dopd"`）与 `casr_elastic`（反事实收益）并排；`casr_static` 等作固定池对照 | **已实现**：两条规则都能单独跑，差异只剩"要不要重解一次 LP 来定价收益" |
| R6 | **平台化编排**（NVIDIA Dynamo、llm-d 一类 worker/router/传输组件化） | worker 生命周期、服务发现、KV-aware routing、可插拔传输 | 模拟器侧的 `PrefillLifecycle` / `ResourceOrchestrator` / `ReconfigExecutor`；真机侧 `casr_control.py` 的 `docker start/stop` 执行器 | **作为底座**，不作为对比算法 |
| R7 | **跨 DC prefill 卸载**（PrfaaS 一类） | 跨数据中心做 prefill，带宽/队列/缓存感知，并调整 P/D 比例 | **`prfaas`**（本地域优先 + 跨域硬预算 `--prfaas-max-offload-ms`）与 `prfaas_tight`（预算 150 ms） | **已实现**：阈值是一条可扫的 CLI 参数，两档臂分别代表"宽松卸载"和"只在便宜时卸载" |
| R8 | **快速冷启动 / 迁移**（ServerlessLLM 一类 checkpoint 加载与迁移；Llumnix 一类实例内迁移） | 缩短实例启动、迁移在途请求 | **`serverless_elastic`**（启动 = 引擎初始化 17.2 s + 权重/带宽，标定到实测冷/热重启）与 **`llumnix`**（队列重平衡 + KV 搬运计费 + 收益/成本闸门） | **已实现**（各有已声明的建模边界：前者只改启动成本，后者只搬"不在飞行批次里"的请求，不做 block 级搬移） |

## 2 我们实现而对照臂没有的

| 能力 | 载体 | 说明 |
|---|---|---|
| 状态条件化的 P-D 关系矩阵 | `CapacityAwareFlowSolver`（LP） | 每个 prefix 类在每台 P 上的命中工作量进入容量与成本，而不是只做命中统计 |
| 结构编辑的反事实评估 | `StructuralEvaluator` + `PrefillLifecycle` | `keep / +P(cold) / +P(warm) / -P` 各重求解一次，按窗口收益与 dwell time 选动作 |
| KV 字节预算作为一等约束 | `shared_links.capacity_bytes_per_s` + 控制器每 tick 的 `_egress_bound_prefill_capacity` | 与"传输代价"分开：一个是硬预算，一个是单位成本 |
| 模型/链路自洽的容量口径 | `hw_service.resolve_runtime_capacities` | prefill 容量 = 执行侧 period 的倒数（含每 step 开销），长度项走实测曲线 |

## 3 公平性修复（2026-09-24）

对照臂与 CASR 臂原先**用的不是同一个价格模型**：`pair_costs` 只在 `--enable-casr` 时构建，
于是非 CASR 臂的 `local_prefill` 决策退回"真机记录常数"（585 ms/1k 同域、1351 ms/1k + 48 ms 跨域），
而那组常数是按**部署的 Qwen3-8B**（147,456 B/token、0.257 GB/s push）标定的。
在压缩 KV 的 P-15B 上这会把搬运高估约 9 倍，导致三条基线臂在 metro 上 **740/740 全走本地重算**，
而 CASR 臂（它建了表）**740/740 走搬运**——两边被不同的模型定价。

修复后（`serving/__main__.py`：`pair_costs` 对所有运行构建），
同一台机器上的决策变成模型感知的：

```text
P-15B / metro   local 355.1 ms  vs transfer  27.5 ms  -> transfer
P-15B / WAN     local 355.1 ms  vs transfer 240.7 ms  -> transfer
Qwen3 / metro   local 355.1 ms  vs transfer 558.6 ms  -> local   (同域 p0->d5)
Qwen3 / WAN     local 355.1 ms  vs transfer 1723.6 ms -> local
```

**这条轴本身就是结论**：同样的 1250-token 请求，压缩 KV 让"搬"在 WAN 上依然划算（240 ms < 355 ms），
而传统稠密模型在 WAN 上只能退回本地重算（1724 ms > 355 ms）。

## 4 2026-09-24：四个缺口都成了可跑的臂

§1 的 R1/R5/R7/R8 此前是"部分/未实现"。现在的状态与**各自的建模边界**：

| 臂 | 语义（一句话） | 建模边界（写进结论里） | 结果 |
|---|---|---|---|
| `dopd` | 利用率 ≥ `scale_out_utilization`（或队列 ≥ 阈值）持续 N tick 就 `+P`；利用率 ≤ `scale_in_utilization` 且全闲才 `-P`；有迟滞与 hold 窗口 | 规则不看目标函数，阈值来自经验，不保证最优 | P-15B WAN 16 rps：4 178 ms / SLO 51.2%，与反事实版 `casr_elastic`（4 129 ms / 55.0%）同档 |
| `prfaas` `/` `prfaas_tight` | 请求先落到自己的域：本地 Prefill 未超阈值就在本地跑；超了才按 `max(transfer, compute wait)` 选，且搬运时间不得超过 `--prfaas-max-offload-ms` | 阈值是全局常数（不是 per-DC 搜索）；不做 P/D 比例调整 | 同档最强：2 447 ms（−26% vs `load`），TTFT p50 1 157 → 446 ms |
| `llumnix` | 每 N ms 比较各 Decode 的**排水时间**，把队首外的请求搬到更空的一台，搬一次扣一次 KV 搬运时间（`migration_ns` 进 TTFT 与总延迟，不改 TPOT），收益不够就不搬 | 只搬"不在飞行批次里"的请求，**不建模** block 级搬移；因此它是 Llumnix 的队列子集 | 本 fabric 上一次都没搬（搬运 240 ms ≫ 队列排水 ~ms）；放松闸门后搬 24 次反而劣化 2 977 → 3 649 ms |
| `serverless_elastic` | 启动 = 引擎初始化 17.2 s + 权重 15.26 GiB / 存储带宽，默认与初始化重叠 | 只改启动成本，不改加载**流程**；带宽是参数 | 20 s → 17.2 s 只值 0.7% 均值；启动不是这一档的瓶颈 |
| `distserve` `/` `distserve_lp` | 枚举拓扑能表达的配比，用 profiler 容量算每档 goodput，取最大者、同分取最少机器，写出集群配置再跑 | 域式拓扑只能 `n:n`；Decode 容量是串行的，需要 `--decode-batch-factor` 标定 | arena 上：未标定 → 选 1:2（实测最差 2 532 ms）；标定 4.8× → 选 2:2（实测最好 1 466 ms） |

**这一轮同时说明了两件事**（建议写进汇报）：

1. **比较的赢家会随 fabric 换人**：这一档最强的是"本地域优先"（PrfaaS 式），
   因为跨域一次搬运 240 ms 而本地重算/本地 Prefill 只要 ~300 ms 量级；
   在 metro 上这个差距会变小，所以每条结论都要带环境。
2. **"均衡队列"不是目标**：Llumnix 臂把请求搬到更空的机器会让**慢卡**更忙，
   在异构步长下反而更差——这与 `deadline_aware_decode`（6.23）和
   `queue_weight`/`tail_weight` 的教训是同一条。

## 5 复现

```bash
# 4 个环境 = {P-15B(类 DSV4), Qwen3-8B(传统稠密)} x {metro, WAN}
# metro：跨域 push 0.33 GB/s / 20 ms RTT，同域 0.77 GB/s
# WAN  ：跨域 push 0.11 GB/s / 48 ms RTT，同域 0.257 GB/s（真机实测）
tests/run_route_arms.py \
  --cluster-config configs/cluster/casr_p15b_3dom_wan_auto.json \
  --dataset workloads/matrix-16rps-slo1500.jsonl \
  --arms load,cache_aware,kv_aware,casr_lp --out-root /tmp/env-wan-p15b
```

结果与判读见 `docs/实验数据集与对比基线说明.md` §6.26。

新臂的复现（每个臂只改一个开关，其余与基线一致）：

```bash
tests/run_route_arms.py --cluster-config <cfg> --dataset <trace> \
  --arms load,prfaas,prfaas_tight,llumnix,casr_elastic,dopd,serverless_elastic
# 配比搜索：先搜 → 写配置 → 再用那个池跑
tests/pd_ratio_search.py --cluster-config <cfg> --dataset <trace> \
  --write-config /tmp/ds.json --verify 3
tests/run_route_arms.py --cluster-config <cfg> --dataset <trace> \
  --arms distserve,distserve_lp --fixed-config /tmp/ds.json
```

数字与判读见 `docs/实验数据集与对比基线说明.md` §6.33；各臂的开关见
`tests/run_route_arms.py::ARMS` / `OVERLAYS`。
