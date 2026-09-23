# 近一年异构 P/D 分离调度的路线，与模拟器里真正跑得起来的基线

本文回答两个问题：**近期（2024-09 起的 [year]）异构 P/D 分离调度研究分几条路线**，
以及**其中哪些机制已经在 `LLMServingSim` 里实现成了可跑的对照臂**、还差什么。

> 版本会漂移：本文按 2026-09-24 的仓库状态写，引用的是**机制**而非具体数字。
> 所有"已实现"都指仓库里能跑的 flag / arm，见文末的复现命令。

## 1 路线划分与覆盖情况

| # | 路线（代表系统） | 该路线的核心机制 | 我们的对照臂 | 覆盖度 |
|---|---|---|---|---|
| R1 | **P/D 解耦与配比**（DistServe、Splitwise 一类的分离式服务） | 把 prefill 与 decode 放到不同实例，按 TTFT/TPOT 预算与 goodput 定 P:D 比例 | `static`（固定池）、`casr_plan3`（等容量固定池）、`casr_static`（单 worker） | **部分**：我们比的是同一个池上的*放置*与*是否重构*；**没有**实现"按 SLO 搜 P:D 比例"的配比搜索 |
| R2 | **前缀感知路由**（SGLang / RadixAttention 一类 cache-aware router） | 最长前缀匹配优先，命中者优先，忙则溢出到最闲 | `cache_aware`（`--request-routing-policy CACHE_AWARE`）：与真机 `_pick_cache_aware` 同语义（含 `CACHE_AWARE_OVERFLOW=0.75` 溢出阈值） | **已实现**（语义对齐真机路由器） |
| R3 | **KV/网络感知路由**（Mooncake / NIXL / LMCache 一类以 KV 为中心的数据面；NetKV 一类网络感知 decode 选择） | 按 KV 传输代价、链路占用、队列选边；把 KV I/O 当作一等成本 | `kv_aware`（`KV_AWARE`）：按 producer 出口排队 + 序列化时间与算力等待取 max | **已实现** |
| R4 | **传输 vs 重算的存算协同**（LMCache/NIXL 的"搬还是不算"决策，真机 `disagg_router.kv_exchange_decision`） | 逐请求比较"搬 KV"与"本地重算 prompt"，两边都带各自的排队外部性 | 所有臂共享：`local_prefill: auto` + `_link_derived_terms`；CASR 臂另加计划 | **已实现**（本次修复后所有臂同价，见 §3） |
| R5 | **动态 P/D 扩缩容**（DOPD 一类按需求/利用率调整实例数） | 需求/利用率超阈值就加实例，低了就删 | `casr_elastic`（`min_active_prefill=1 → max`，阈值来自 `StructuralEvaluator` 的反事实收益）；`casr_static`/`hetero6` 等固定池作对照 | **部分**：我们实现的是"反事实收益 > 启动成本"的规则，**没有**实现纯利用率阈值版本作为单独臂 |
| R6 | **平台化编排**（NVIDIA Dynamo、llm-d 一类 worker/router/传输组件化） | worker 生命周期、服务发现、KV-aware routing、可插拔传输 | 模拟器侧的 `PrefillLifecycle` / `ResourceOrchestrator` / `ReconfigExecutor`；真机侧 `casr_control.py` 的 `docker start/stop` 执行器 | **作为底座**，不作为对比算法 |
| R7 | **跨 DC prefill 卸载**（PrfaaS 一类） | 跨数据中心做 prefill，带宽/队列/缓存感知，并调整 P/D 比例 | 我们的 WAN 环境（跨域 push 0.11 GB/s + 48 ms RTT）上的全部臂 | **部分**：环境有了，"跨 DC 阈值搜索"本身没有单独实现 |
| R8 | **快速冷启动 / 迁移**（ServerlessLLM 一类 checkpoint 加载与迁移；Llumnix 一类实例内迁移） | 缩短实例启动、迁移在途请求 | 只做了**启动成本建模**（`startup_ms`/`warmup_ms`，6.25 用 46 次真机重启标定）；**没有**实现迁移臂 | **未实现**（明确的空白） |

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

## 4 复现

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
