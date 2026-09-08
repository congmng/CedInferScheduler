# CASR 调度算法实施设计

## 1. 目标与第一版边界

CASR（Cache-aware Affinity and Structural Reconfiguration）的目标是：在跨域 Prefill/Decode（P/D）分离系统中，先为**当前结构**求得面向请求类型的最优 P-D 分配；再判断增加、删除或替换一个 Prefill 实例是否能在一个明确的决策窗口内产生正的净收益。

第一版只做以下决策：

- 固定 Decode 实例集合，不做 D 扩缩容和 active Decode KV 迁移；
- 对新到请求做 P-D affinity，不重排正在 Decode 的请求；
- 每轮最多编辑一条 P 行：`keep`、`+P@domain(cold)`、`+P@domain(warm)` 或 `-P_i`；
- Prefix 不做全局 placement 或 eviction 优化，只作为 P 的有效供给和 warmup 代价的输入；
- 使用滑窗/EWMA，不引入预测模型或 RL。

因此核心链路是：

```text
状态快照 → 按 prefix class 构造流量/代价 → 固定结构最小成本流
        → 生成少量 P 行编辑候选 → 反事实重求解 → 窗口净收益
        → keep / reconfigure，并下发 affinity
```

## 2. 为什么要按请求类求流

只有总流量矩阵 `f_ij` 不足以表达 Prefix 的作用：不同请求的输入长度、KV 大小、命中长度和 Decode 负担不同。控制器把新到请求聚合为有限个请求类 `k ∈ K`，决策变量为：

`f_ijk`：在当前控制窗口内，从 Prefill P_i 产生、交给 Decode D_j、且属于 prefix class k 的预计请求率（req/s）。

一个类至少含有：

```text
class_id, model_id, prefix_id, input_tokens, output_tokens,
kv_bytes_per_request, arrival_rate, slo_ttft_ms, slo_tpot_ms
```

`prefix_id` 不保存原文，可由“模型 ID + 前 L 个完整 KV block 的最后一个链式 hash + prefix token 长度”构成。这样既与 LLMServingSim/vLLM 的 block cache 对齐，也能将有相同可复用前缀的请求归为一类。

聚合规则：在最近窗口 `W_obs` 内按 `(model_id, prefix_id, input/output bucket)` 聚类；低频类归入 `OTHER`，保证 `|K|` 有上限（第一版建议 16--64）。

## 3. 调度状态 S_t

慢层每 `T_ctrl`（建议 10 s）或发生 SLO/链路事件时读取一次状态。

| 状态 | 粒度 | 来源 | 用途 |
|---|---|---|---|
| `λ_k` | class | 最近窗口到达率 EWMA | 流量需求 |
| `r_k` | class | reuse EWMA/近期到达 | warm 是否可摊销 |
| `H_ik` | P,class | 命中 token 长度 EWMA | P 的有效 Prefill 工作量 |
| `B_ik` | P,class | 缓存块或字节数 | warmup 代价与可用性 |
| `qP_i`, `qD_j` | instance | waiting/running/remaining work | 排队风险 |
| `μP_i,k` | P,class | profile + 命中状态 | P 行供给 |
| `μD_j` | D | profile + active Decode residual | D 列供给 |
| `RTT_ij`, `BW_ij` | link | 网络测量/仿真参数 | KV 转移时间 |
| `A_ij` | pair | 拓扑、模型、SLO、连接器能力 | 可行性掩码 |

### 3.1 从 LLMServingSim 提取的基础状态

LLMServingSim 已具备请求 token IDs、链式 block hash、每请求命中 token 和每个 pool 的 `block_hash → block` 索引。原生输出只有累计 hit ratio，不能直接产出 class 热度，因此新增 `PrefixProfiler` 时应在请求到达、cache lookup、cache insert 和 block eviction 时更新聚合表。

每轮对每个 P 实例导出：

```text
prefix_id → {request_count, reuse_ewma, hit_tokens_ewma,
             cached_blocks, cached_bytes, last_access_ns}
```

`last_access_ns` 和衰减的 `reuse_ewma` 可防止已经冷却的旧热点被误判为 warm 候选。

## 4. 有效容量和代价模型

### 4.1 Cache-aware Prefill 容量

对 P_i 服务 class k，令：

`effective_tokens_ik = max(1, input_tokens_k - H_ik)`。

通过离线 profile 表查询相同 GPU、batch/context bucket 下每个有效输入 token 的 Prefill 时间 `tP_i(effective_tokens_ik)`，得到：

`μP_i,k = utilization_budget_i / tP_i(effective_tokens_ik)`。

这里的 `utilization_budget_i` 是可用于新请求的时间份额；应扣除当前 running/queued Prefill 工作，不能用 GPU 理论峰值替代。

Decode 容量使用剩余能力：

`μD_j = μD_profile_j - active_decode_work_j`。

若不同 class 的输出长度相差明显，应将 D 容量约束改写为工作量单位：`f_ijk × expected_output_tokens_k`，而不是简单 req/s。

### 4.2 Pair 代价

对每个可行三元组，使用预计 TTFT 风险作为单位请求代价：

`c_ijk = tP_i(effective_tokens_ik) + rtt_ij + kv_bytes_k / bw_ij + qD_j + tD_j(k)`。

第一版允许 `qD_j` 使用观测队列等待时间或简单的排队近似。若预测 TTFT 超过 class SLO，则增加稳定的大惩罚 `p_slo`，而不是把 pair 直接删掉；只有拓扑、模型或 connector 不兼容才令 `A_ij=0`。

共享 WAN 会竞争时，补充链路约束：

`Σ_(i,j,k ∈ link l) f_ijk × kv_bytes_k ≤ BW_l × η_l`。

`η_l` 是保守利用率系数，第一版可取 0.7--0.8。

## 5. 固定结构的内层优化

给定结构 S，解带 overflow sink 的线性规划/最小成本流：

```text
min  Σ_i,j,k f_ijk c_ijk + Σ_k u_k p_drop,k

s.t. Σ_j f_ijk ≤ μP_i,k                         ∀ i,k
     Σ_i f_ijk ≤ μD_j                            ∀ j
     Σ_i,j f_ijk + u_k = λ_k                     ∀ k
     Σ_(i,j,k ∈ link l) f_ijk kv_bytes_k ≤ BW_l η_l  ∀ l
     f_ijk = 0                                    if A_ij = 0
     f_ijk, u_k ≥ 0
```

`u_k` 是未服务/超载流量。它使候选结构在供给不足时仍可比较，不会因“无可行解”让 planner 失效。`p_drop,k` 取远大于正常 SLO 违约代价的值。

输出包括：`F*(S)`、单位时间目标 `J_rate(S)`、每个 P-D-class 的 affinity 比例，以及 overflow、D 利用率和链路利用率。

当实现库不方便表达 link 约束时，第一版可先用 OR-Tools LP；没有共享链路约束的纯 bipartite 情况可退化为 NetworkX/OR-Tools min-cost flow。

## 6. 诊断只用于触发

每轮都重求一个小 LP 是可接受的；只有在以下信号出现时才展开反事实候选：

- `overflow_ratio > 0` 或任一高优先级 class 的 SLO 风险高；
- 高成本边流量比例 `E_exp` 超过阈值；
- assignment regret 高：实际加权代价显著高于各 P 对该 class 的可达最小 pair cost；
- D 或 WAN 利用率接近饱和，而其他可用 D/P 仍有余量；
- Prefix 热点排序发生变化（例如 top-K 的 Jaccard similarity 低于阈值）；
- 当前结构的最小驻留时间已到，且负载显著下降，允许评估 scale-in。

这些指标绝不直接决定扩缩容。最终动作始终由反事实后的净收益决定。

## 7. 候选动作与反事实状态

候选空间刻意保持很小：

| 动作 | 新行状态 | 成本/收益来源 |
|---|---|---|
| `keep` | 不变 | 基准 |
| `+P@d(cold)` | 新 P，无可用 Prefix | startup、较低初始 `H_new,k` |
| `+P@d(warm)` | 新 P，预取 Top-K prefix | startup + warm I/O/wait；较高 `H_new,k` |
| `-P_i` | 删除 P_i | 节省资源，但丢失该行容量和局部缓存价值 |
| `relocate(P_i,d)` | `-P_i + P@d(cold/warm)` | 第一版可作为两个连续动作，后续再合并 |

新增位置仅从以下来源生成：高成本流量目的 D 所在域、低利用强 D 所在域、当前热门 Prefix 的存储可达域，以及预留的 P worker pool。每个域只保留最多一个 cold 和一个 warm 候选。

### 7.1 warm 选择的 Top-K

对候选域 d，按下面的近似价值选 prefix：

`warm_value_k = r_k × saved_prefill_time_k × W_eval - load_bytes_k / storage_bw_d`。

按 `warm_value_k / load_bytes_k` 贪心选择，直到 warm 预算 `B_warm` 或预热时间 `T_warm_max` 用尽。随后用所选集合重新计算新行 `H_new,k`、`μP_new,k` 与 warm 成本。

这不是全局 cache placement；它只回答“已决定在此处加 P 时，要不要预热及预热哪些热点”。

## 8. 用时间窗口统一 Structural Gain 单位

`J_rate` 的单位是“预计代价/秒”，而启动和预热是一次性成本，不能直接相减。对动作 a 使用评估窗口 `W_eval`：

`G_W(a) = W_eval × [J_rate(S) - J_rate(S_a)] - C_start(a) - C_warm(a) - C_drain(a) - C_cache_loss(a)`。

其中：

- `W_eval`：从新行可服务开始，到下一次允许结构变化前的时间；第一版可取 `max(T_ctrl, dwell_time_remaining)`；
- `C_start`：P 从 admission 到 ready 的等待及资源成本；仿真第一版可用固定 profile 值；
- `C_warm`：预取时间、占用的共享带宽成本和 warmup 期间少得到的服务收益；
- `C_drain`：停止 admission 并等待该 P 已有请求清空的成本；
- `C_cache_loss`：scale-in 后预计重算 prefix token 的成本。

选择 `argmax_a G_W(a)`。只有同时满足下面条件才执行：

```text
G_W(a*) > θ_abs
G_W(a*) / max(W_eval × J_rate(S), ε) > θ_rel
从上次结构动作起已过 dwell_time
执行动作不会使 active Decode 请求迁移或失效
```

若 warm 尚未完成，P 处于 `WARMING`，不能接收新请求；warm 失败或超时则退化为 cold，不重复触发结构动作。

## 9. 两时间尺度执行

### 快层：每请求

慢层对每个 `(P_i, k)` 下发 D affinity 分布：

`π_ijk = f_ijk / Σ_j f_ijk`。

请求到达时：

1. 根据 prefix class 选择可接收该类、且 `μP` 残余最大的 P；
2. P 完成后只在 `π_ijk > 0` 的 D 集合内选当前边际成本最低者；
3. 若目标 D 临时满载，可在同一 affinity 集合内降级选择；若均不可用，进入 overflow/等待队列。

快层不重新运行 LP，也不改变结构。它只对遥测误差和瞬时队列波动做局部纠偏。

### 慢层：控制周期

```text
snapshot = collect_state()
classes  = aggregate_recent_requests(snapshot, max_classes=64)
base     = solve_flow(snapshot.structure, classes)

if not should_evaluate(base, snapshot):
    publish_affinity(base)
    return KEEP

candidates = generate_candidates(snapshot, base)
for action in candidates:
    counterfactual = apply_row_edit(snapshot, action)
    result[action] = solve_flow(counterfactual.structure, classes)
    gain[action] = window_gain(base, result[action], action, snapshot)

best = argmax(gain)
if admissible(best, snapshot):
    execute(best)                 # admission/drain/warm，不触碰 active D KV
publish_affinity(best.result if executed else base)
```

规划复杂度为 `O(|candidates| × LP(|P|×|D|×|K|))`。在 `|P|≤16, |D|≤16, |K|≤64, candidates≤12` 时，控制周期 10 s 内有足够余量；每轮还应记录 solver 时延和候选数。

## 10. LLMServingSim 接入设计

当前 LLMServingSim 已能模拟 P/D handoff，但 `Router.transfer_prefill_request()` 只按 RR/LOAD 在 Decode pool 选择目标，且现有 cluster config 的 `link_bw/link_latency` 是拓扑级参数。CASR 不应改动 attention、KV block 或 ASTRA-Sim 内核，而应新增控制层和 router policy。

建议新增文件：

```text
serving/casr/
  state.py          # Snapshot、PrefixClass、WorkerState 数据类
  prefix_profiler.py# prefix_id、热度、H_ik、缓存驻留统计
  capacity.py       # μP_i,k、μD_j
  cost.py           # A_ij、c_ijk、link constraints
  flow.py           # OR-Tools LP wrapper
  evaluator.py      # 候选生成、warm Top-K、G_W
  controller.py     # 每 T_ctrl 触发、下发 affinity
```

最小侵入点：

| 位置 | 修改内容 |
|---|---|
| `serving/core/router.py` | `CASR` 路由策略；在 P 完成时根据 controller 已下发的 `π_ijk` 选 D |
| `serving/core/request.py` | 增加 `prefix_id`、`class_id`、实际 hit/compute/warm 观测字段 |
| `kv_cache_manager.py` | 仅增加 lookup/insert/evict observer，不改变块管理语义 |
| `serving/__main__.py` | 每控制周期调用 controller；将 P worker 标记为 `ACTIVE/WARMING/DRAINING/INACTIVE` |
| cluster config | 增加 domain、候选 P pool、pair/link 属性；先用模拟参数表达多域 |
| 输出 CSV/JSON | 写入 request 的 class、P/D、命中 token、KV bytes、决策版本；每周期写状态快照和动作日志 |

第一阶段用“预先创建但可 admission/drain 的 P 实例”模拟 scale-out/scale-in。这样可先验证结构算法，避免把 Kubernetes 生命周期和系统论文变量混入第一版。真实 start/warm 时间在第二阶段替换为测量值。

## 11. 参数初值与安全规则

| 参数 | 建议初值 | 说明 |
|---|---:|---|
| `T_ctrl` | 10 s | 慢层周期 |
| `W_obs` | 30--60 s | 热度 EWMA 观测窗 |
| `W_eval` | 60 s | 结构收益摊销窗 |
| `dwell_time` | 60--120 s | 两次结构编辑最小间隔 |
| `max_classes` | 32 | 先保证模型小且可解释 |
| `max_actions` | 12 | 每域 cold/warm + scale-in |
| `B_warm` | 可用 cache 的 10--20% | 预热预算 |
| `η_link` | 0.7 | 链路保守可用比例 |
| `θ_rel` | 3--5% | 排除微小噪声收益 |

安全规则：

- active Decode KV 永不迁移；
- `DRAINING` P 停止接新请求，清空后才退出；
- `WARMING` P 未达到可用阈值前不接请求；
- 遥测缺失时将 link 视为保守低带宽、cache hit 视为 0；
- LP/求解失败时保持上一版 affinity，不执行任何结构动作；
- 每个动作与其反事实输入、求解结果、`G_W` 分项持久化，保证可复现与论文可解释性。

## 12. 实施与验收顺序

1. **可观测性**：写出 request 级 `class_id/P/D/hit_tokens/kv_bytes` 与每周期 prefix 热度快照。验收：可画出 `H_ik` 和 top-K 热度随时间变化。
2. **固定结构分配**：实现 `f_ijk` LP 和 `CASR` affinity router，不改变实例数。验收：Case A 中优于 nearest-D/LOAD。
3. **缓存有效供给**：将 `H_ik` 接入 `μP_i,k`。验收：请求率不变而热点改变时，分配与容量估计随之变化。
4. **行编辑反事实**：实现 cold、warm、scale-in 与 `G_W`、dwell time。验收：总 P 容量充足但拓扑失配时仍能做有正收益的 +1 P；热度/BW 改变时 warm/cold 排序翻转。
5. **多域与真实校准**：扩展 cluster/link 配置，使用真实 vLLM/LMCache profile 替换启动、缓存加载与链路参数。验收：仿真趋势与小规模真实实验方向一致。

## 13. 不应在第一版加入的内容

- P 和 D 双边全局扩缩容；
- active Decode KV migration；
- Prefix 全局副本放置、复杂 eviction 或 KV 压缩；
- 基于深度预测/RL 的策略；
- 以 cache hit rate 作为独立优化目标。

第一版的可检验主张应始终保持为：**Prefix 状态会改变 P 行的真实有效供给和 warm 行形成代价；在给定 P/D 结构内求得最优分配后，只有当编辑一条 P 行在统一窗口内降低全局净成本时，才进行重构。**
