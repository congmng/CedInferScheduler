# 基于 vLLM 与 LMCache 的存算协同结构调度方案

## 1 研究定位

**基座不是贡献，控制器才是贡献。**

vLLM 负责高性能 LLM serving、Prefix Caching、KV Connector、worker 指标和 benchmark；LMCache 负责 KV 的外部存储、跨实例复用和远程加载/写回。研究系统在它们之上增加一个独立的控制平面，暂定名称为 **CASR Controller**（Cache-aware Affinity and Structural Reconfiguration）。

系统解决的不是“怎样把 KV 放到远端”，而是：

> 在 P 的真实供给被 Prefix 状态改变、D 的能力和网络位置不同的条件下，何时应当复制状态并增加、删除或迁移一个 P，使 P-D 关系的整体重构收益大于资源与状态迁移成本？

这使研究与 vLLM/LMCache 本身形成清晰边界：二者提供机制，CASR 决定是否、何时、在哪里使用这些机制。

## 2 系统架构

```text
                        慢时间尺度 5--30 s / 事件触发
 ┌──────────────────────────────────────────────────────────────────┐
 │                        CASR Controller                           │
 │ StateCollector -> CapacityEstimator -> FlowMatcher               │
 │                    -> Counterfactual Evaluator -> Executor       │
 └─────────────┬──────────────────┬──────────────────┬──────────────┘
               │ affinity         │ warm/cold        │ +1/-1 P
               ▼                  ▼                  ▼
      ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐
      │ vLLM P workers │  │ LMCache backend │  │ vLLM D workers │
      │ Prefix Cache   │  │ remote KV store │  │ KV Connector   │
      └───────┬────────┘  └────────┬────────┘  └────────▲────────┘
              │ Prefill KV          │ hot-prefix load     │ Decode KV
              └─────────────────────┴─────────────────────┘

                        快时间尺度 请求级
 Request -> ingress/router -> selected P -> affinity-constrained D
```

### 2.1 基座分工

| 层次 | 使用组件 | 不需要重新实现的能力 | 本研究的接入点 |
|---|---|---|---|
| 推理执行 | vLLM | continuous batching、attention、token generation、请求生命周期 | P/D worker 的角色、metrics 和流量入口 |
| 本地 KV | vLLM Prefix Caching | block/hash、命中、分配和 eviction 基础机制 | 提取请求级 hit prefix length 与 cache occupancy |
| 外部 KV | LMCache | KV 存取、远端后端、vLLM connector、传输计时 | warm P 前缀选择、写/读字节数、加载耗时 |
| 状态采集 | Prometheus / vLLM metrics / 自定义 exporter | queue、throughput、latency、GPU 监控 | 合成调度状态 S_t |
| 本研究 controller | Python 服务 | 无 | 容量估计、矩阵匹配、结构收益和执行策略 |

### 2.2 两时间尺度

**快层（每请求）**：在慢层给出的 P-D affinity 集合中，按 D queue、残余容量和当前可用带宽选择具体 D。第一版不把快层作为创新，只需复用/轻量实现一个可解释的 least-cost 选择。

**慢层（5--30 秒，或事件触发）**：只处理新增、删除、迁移一个 P，以及 cold/warm 形成方式。慢层读取 LMCache 和 vLLM 的状态，做小候选集合的反事实评估，避免在线全局 MILP/RL。

## 3 需要实现的六个模块

| 模块 | 输入 | 输出 | 预计实现量 |
|---|---|---|---|
| `StateCollector` | vLLM queue/latency/throughput、LMCache 命中与 I/O、RTT/BW、worker 状态 | 状态快照 S_t | 小 |
| `PrefixProfiler` | 请求 prefix ID、hit length、LMCache read/write | 每个 P 和 prefix class 的 H_ik、热度、复制代价 | 小 |
| `CapacityEstimator` | GPU profile、ISL/OSL、H_ik | cache-aware μP_i、residual μD_j | 小 |
| `CostMatrixBuilder` | KV size、BW/RTT、D queue、decode profile | A、C、共享链路约束 | 中 |
| `FlowMatcher` | A、C、μP、μD、λ | F*(S)、J*(S)、affinity | 小；最小成本流库即可 |
| `StructuralEvaluator` | 当前解、cold/warm/scale-in 候选 | G(a)、选中动作与理由 | 中 |

`ReconfigExecutor` 是上述模块的薄封装：调用 worker/orchestrator 新建或 drain P，并在 warm 动作中请求 LMCache 预取 Top-K 前缀。第一版可以将 P worker 预先启动，通过 admission/drain 模拟启停，减少 Kubernetes 改造量；实验成熟后再接 Ray 或 Kubernetes 真实弹性。

## 4 核心创新点

### C1 状态条件化的 P-D 关系矩阵

以往的 P-D 分离调度往往把 P 看作单一供给、把 D 看作单一服务端；本工作将关系显式建模为 P、D、请求类三元组：

`C_ijk(t) = T_prefill(i,k,H_ik) + T_KV(i,j,k,BW,RTT) + Q_D(j) + T_decode(j,k)`。

这里 `H_ik` 是 P_i 对请求类 k 的可复用 Prefix 覆盖率。它不是把 cache hit rate 简单加到总分，而是同时改变 P_i 对该类请求的 Prefill 工作量、有效行容量和 P-D 偏好。

**可验证差异**：同样的 GPU、同样的平均负载，但 Prefix 热点位置不同，应得到不同 P-D affinity 和扩容位置。

### C2 结构收益驱动的 P 重构

将 `+1/-1/relocate P` 解释为对 P-D 关系矩阵的增行、删行或替换行。对候选动作 a，不依赖“GPU 利用率超过阈值”做决定，而是反事实重求分配：

`G(a|S_t) = J*(S_t) - J*(S_t^a) - C_reconfig(a)`。

仅当 `G(a)>θ`、SLO 可行且满足最小驻留时间时执行。这捕捉一个传统 autoscaler 易遗漏的情况：**总 P 容量已够，但现有 P 行无法低成本连到有剩余能力的 D，改变一行仍可带来净收益。**

**可验证差异**：Case B 中总 μP≥λ；Load-based autoscaler 不扩，CASR 仍能因消除高成本边而选择特定域的 +1 P，并改善 P95 TTFT/跨域流量。

### C3 面向结构收益的 cold/warm 状态形成决策

新增 P 在实例 ready 时并不等于拥有完整有效算力。对同一候选位置 k，控制器分别构造：

- `+P@k(cold)`：无 Prefix 预取，启动成本低但前期 μP 低；
- `+P@k(warm)`：从 LMCache 读取 Top-K 高价值 Prefix，产生读流量和等待，但提高短窗口有效 μP。

只比较 `G_warm` 与 `G_cold`，而不是独立最大化 cache hit rate。前缀 k 的轻量价值为：

`V_k = predicted_reuse_k × saved_prefill_time_k - LMCache_load_time_k`。

**可验证差异**：前缀热度或存储链路变化会翻转 warm/cold 的动作排序；报告 `Effective Capacity Ramp-up Time`、`recomputed prefix tokens`、warm I/O 和 `G_warm-G_cold`。

### C4 增量连续性与最低扰动

只对当前周期的新到流量求 F_t，不迁移正在 Decode 的 active KV；D 的已有活跃请求被扣成 residual capacity。再用 `|ΔP|≤1`、dwell time 和收益阈值限制结构变化。

这不是单独的大创新，但使方案能部署在 vLLM + LMCache 上，并能和激进的全局重排形成有价值的工程对照。

## 5 与近期系统和 SOTA 的比较

以下将“近期 SOTA”分为四条路线。项目/论文版本演进很快，正式写作时应固定仓库 commit 与论文版本；比较的重点是问题边界，而不是笼统宣称性能优于所有系统。

| 系统/路线 | 强项 | 已解决问题 | 尚缺的点 | CASR 的明确区别 |
|---|---|---|---|---|
| **vLLM + LMCache** | 成熟推理、Prefix Cache、外部 KV 复用 | 如何高效执行/存取/复用 KV | 不决定 P 的跨域位置、P-D 流量矩阵或结构重构是否值得 | 以其作为 substrate，新增 cache-state-to-capacity-to-reconfiguration 控制闭环 |
| **SGLang / RadixAttention** | 高效 prefix reuse、请求调度、PD serving 能力 | 高命中率和本地请求执行 | 缺少以缓存状态驱动 P 行增删的反事实全局评估 | 不只做命中友好路由，而是比较 warm/cold 后的全局净收益 |
| **Mooncake / KVCache-centric serving** | 分布式 KV、Transfer Engine、KV-aware 数据路径 | 如何让 KV 跨节点流动得更快 | 通常不以“当前 P-D 结构是否值得重构”为主目标 | 将 KV I/O 作为 C_reconfig 的测量项，决定是否启动传输 |
| **NVIDIA Dynamo** | 生产化 P/D 分离、KV-aware routing、NIXL 与编排 | worker/router/传输层组件化 | 通用平台不等于面向 prefix 状态的结构收益算法 | 可作为工程 SOTA；CASR 提供可插拔的慢层策略 |
| **DistServe / Splitwise** | P/D 解耦与资源配比 | 分离 prefill/decode 的吞吐、tail latency、资源配置 | 原始目标多为实例配比/批处理；缺少外部 Prefix 状态对扩容有效能力的建模 | 以 cache-aware μP 与跨域 pair cost 扩展 P/D 配置问题 |
| **DOPD 类动态 P/D autoscaling** | 根据 workload 调整 P/D 比例 | P/D 数量随负载变化 | 通常以需求/利用率为主，且不显式比较结构重构的净收益 | capacity sufficient 仍可重构；行操作价值来自矩阵重求解 |
| **NetKV 类网络感知 Decode 选择** | 已有 P 完成后选择 D，考虑 queue/网络/KV 传输 | 固定 P/D pool 内的 request-level D selection | 不改变资源结构，难消除由拓扑造成的长期高成本边 | CASR 使用相同网络信号，但在慢层改变 P 行和可行域 |
| **PrfaaS 类跨数据中心 Prefill 卸载** | 跨 DC prefill、带宽/队列/缓存感知路由与资源调整 | 远端 prefill 是否值得，P/D 比例调整 | 若只说跨 DC + cache + 弹性，容易与其重叠 | 主张应收缩为“状态条件化 P-D 关系矩阵 + 最小结构重构收益”，而非泛泛跨 DC 扩缩容 |

### 5.1 必须避免的过度宣称

- 不要说“首次使用 vLLM/LMCache 做 P/D 分离”。基座能力会快速演进，也不是算法贡献。
- 不要只宣称“联合考虑缓存、网络、异构 GPU 和扩缩容”。PrfaaS、Dynamo、Mooncake 等方向会造成明显重叠。
- 不要把 `cache hit rate` 当作主要胜利指标；这容易被认为只是缓存工程。
- 不要把所有 active Decode KV 迁移、全局 cache placement、D 扩缩容、复杂预测/RL 一次性加入第一版。

### 5.2 建议的论文主张

> **Prefix 状态不是被动的缓存统计，而是决定 Prefill 实例真实供给和结构重构成本的运行时状态。** 因此，跨域 P/D 弹性不应只按负载触发，而应比较 P-D 关系结构改变前后的全局净收益；远程 KV 加载仅在它能提高该净收益时被执行。

这个主张同时区分了：

1. 仅有 KV 系统（能搬状态，但不知道何时值得搬）；
2. 仅有 router（能在既有结构中选边，但不能改变结构）；
3. 仅有 autoscaler（能增减实例，但不能识别缓存和网络导致的结构性失配）。

## 6 基线、消融和证据链

### 6.1 公平基线

所有方法应使用相同的 vLLM、LMCache、模型、P/D workers、Prefix Cache 容量、KV backend 和内层 matching solver。唯一差异是控制策略，避免把基础设施差异误归因于算法。

| 方法 | 使用 LMCache | P-D matching | P 重构规则 | 证明什么 |
|---|---:|---:|---|---|
| B0 Static | 是 | 相同 min-cost flow | 不重构 | 固定结构的上限/下限 |
| B1 LoadScale | 是 | 相同 min-cost flow | utilization 或 demand gap | 结构收益优于容量阈值 |
| B2 NetworkHeuristic | 是 | 相同 min-cost flow | 在高成本流量域 +1 P | 反事实全局评估优于局部启发式 |
| B3 SG-NoState | 是 | 相同 min-cost flow | Structural Gain | cache state 不进入 μP 与 warm/cold，隔离 C1/C3 |
| Ours CASR | 是 | 相同 min-cost flow | state-aware Structural Gain | 完整贡献 |

### 6.2 核心消融

| 消融 | 删除内容 | 应观察到的现象 |
|---|---|---|
| `NoCacheCapacity` | 用 GPU 理论吞吐替代 μP | 热点漂移后误判容量，TTFT 和重算 token 上升 |
| `NoWarmCounterfactual` | 永远 cold 或固定 warm | 不同 prefix 热度/BW 下存在无效预热或恢复变慢 |
| `NoStructuralGain` | 按利用率阈值扩缩 | Case B 中错过结构性重构，或 Case C 中产生多余动作 |
| `NoNetwork` | C_ij 去掉 RTT/BW | WAN 恶化时错误选择远端 D/P |
| `NoHysteresis` | 去掉 θ/dwell time | 动态 trace 中配置翻转与无效 I/O 增多 |

### 6.3 需要交出的关键图

1. **结构失配案例图**：总 μP 足够，但 Static/LoadScale 仍有高成本 P-D 边；CASR 的 +1 P 改善 J、P95 TTFT 和跨域 KV 流量。
2. **cold/warm 决策相图**：横轴 prefix reuse，纵轴 LMCache load bandwidth 或 warmup cost，颜色为 `argmax(Gcold,Gwarm,keep)`。
3. **动态时间线**：请求率、热点变化、J、active P、warm/cold 动作、P95 TTFT 同图，解释决策时间点。
4. **存算协同定量图**：recomputed prefix tokens、saved Prefill GPU time、Effective Capacity Ramp-up Time、warm I/O 与 SLO goodput。
5. **在线开销图**：候选数 vs planner latency；证明小邻域评估可在控制周期内完成。

## 7 最小实施计划

### 第 1 阶段 固定 P/D 下状态和分配

- 部署 vLLM + LMCache，两域各至少 1 个 P、1 个 D；若真实 GPU 不够，可先用逻辑域/限速网络。
- 固定资源结构，采集 cache hit、LMCache 读写、queue、KV size、RTT/BW 和 decode profile。
- 做 `CostMatrixBuilder + FlowMatcher`；完成 Static、nearest-D、queue-only 的对比。

**验收**：能在 Case A 中输出可解释矩阵并降低 P95 TTFT 或 P-D cost。

### 第 2 阶段 Cache-aware capacity

- 将请求按 prefix class 聚合，计算 hit prefix length 和 `effective_tokens`。
- 对每类 GPU/输入长度建立 prefill、decode profile 表，得到 μP 和 μD。
- 完成 `SG-NoState` 与 `NoCacheCapacity` 对照。

**验收**：在热点变化时，可证明相同实例数的有效供给不同。

### 第 3 阶段 Structural Gain 和 warm/cold

- 先将候选限制为 `keep/+1 P@edge(cold)/+1 P@edge(warm)/+1 P@cloud(cold)/+1 P@cloud(warm)/-1 P`。
- 对每个候选重求最小成本流并计算 G；把 warm 的 LMCache load 时间和字节数计入 `C_reconfig`。
- 通过 P 预启动 + admission/drain 实现安全的执行闭环。

**验收**：Case B/D/E 下的决策与预期结构性收益一致，且无频繁翻转。

### 第 4 阶段 扩展和论文评估

- 使用 trace-driven simulator 扩大 GPU/域规模，真实系统提供参数和小规模可信验证。
- 运行 Case A--F、全部基线与消融，报告置信区间。
- 固定 vLLM、LMCache、CUDA、模型和网络配置版本。

## 8 版本与复现建议

- 记录 vLLM commit/tag、LMCache commit/tag、CUDA、PyTorch、模型版本、GPU 驱动和 KV backend。
- 先用 LMCache 的内存/本机远端后端验证机制，再将后端切换为跨域可观测的 Redis、磁盘或对象存储；实验结论要区分“逻辑 WAN 限速”与“真实网络”。
- 必须确认所选 vLLM/LMCache 版本的 Connector API 与 PD disaggregation 接口相容。若接口版本不兼容，保持 vLLM serving 和 LMCache Prefix reuse，P-D KV 传输先用框架原有 connector；不要为了统一接口改动 attention/KV block 内核。
- 正式和 SOTA 比较时，优先复现控制策略（Static、LoadScale、network-aware routing），而不是承诺复现每个大型系统的全部代码路径。

## 9 与最近推理调度的实质差异

### 9.1 先区分调度问题所在的时间尺度

近期工作并不是在解决同一个问题。可以按决策时间尺度分成四类：

| 调度类型 | 典型代表 | 决策对象 | 能否改变 P-D 结构 | 是否把 Prefix 状态转成计算供给 |
|---|---|---|---:|---:|
| 请求级路由 | NetKV、KV-aware router、Mooncake router | 当前请求选哪个 D/哪个 KV 副本 | 否 | 通常只用于估算传输量或命中 |
| KV 数据路径 | LMCache、Mooncake、NIXL | KV 放哪、何时取、如何传 | 通常否 | 体现为存取代价，不负责资源重构 |
| 容量/比例弹性 | DistServe、Splitwise、DOPD 类 | P/D 数量、比例或实例副本 | 部分 | 多按 λ、队列、利用率、KV occupancy |
| 生产编排平台 | Dynamo、llm-d | worker、路由、拓扑和生命周期 | 能，但策略较通用 | 由平台指标驱动，不专注 prefix 价值 |
| **CASR** | 本方案 | P-D 流量矩阵 + P 行增删/迁移 + cold/warm 形成 | **是，核心对象** | **是，直接进入 μP 和 C_reconfig** |

因此，本方案不是再提出一个 request-level router，也不是重新实现 LMCache 的远程 KV，而是在它们之上增加一个**慢时间尺度的结构调度层**。

### 9.2 与常见近期方法的逐项区别

**与 NetKV/KV-aware routing 的区别。** 这类方法在 P 已经确定、D pool 已经确定的前提下，为每个完成 Prefill 的请求选择代价最低的 D。它可以减少当前请求的网络和队列代价，但不能解决“所有 P 都只能连接到一个拥塞/远端 D”的结构瓶颈。CASR 复用相同的 queue、RTT、BW 信号，但进一步评估新增一条 P 行是否能改变全局最优流量分配。

**与 LMCache/Mooncake 的区别。** 这些系统解决“KV 如何高效存取和传输”，CASR 解决“是否值得为一次结构重构付出这次 KV 复制/预热成本”。同一个 warmup 操作在短期负载、热点稳定且带宽充足时可能值得，在热点漂移或 WAN 拥塞时则可能不值得。区别不在于能否复制 KV，而在于复制决策是否受全局 P-D 结构收益约束。

**与 DistServe/Splitwise/DOPD 类 P/D 弹性的区别。** 这类工作主要依据请求率、输入/输出长度、队列或利用率调整 P/D 比例。CASR 的关键反例是：即使 `Σ μP_i ≥ λ`，仍可能因为 P-D 拓扑和 D 列容量不匹配而产生大量高成本边；此时需要结构性扩容，而不是等待利用率超阈值。反过来，负载下降时也不应简单删除利用率最低的 P，而应删除结构价值最低且 cache loss 最小的行。

**与 Dynamo/llm-d 等平台的区别。** 这些平台提供很强的 worker 编排、服务发现、路由和传输基础设施，但它们是通用平台，不等于本文提出的具体策略。CASR 可以作为其上的一个可插拔 planner：平台负责执行 `scale/drain/start`，CASR 负责给出“在哪个域增加哪种 P、cold 还是 warm、为什么值得”。

**与 PrfaaS 类跨数据中心 Prefill 调度的区别。** 两者都涉及跨域、Prefill、Prefix 和带宽，因此这是最需要正面区分的相邻方向。CASR 不把主要问题表述为“远端 Prefill 卸载阈值”或“跨 DC P/D 比例二维搜索”，而是把当前 P-D 对应关系显式写成状态条件化矩阵，研究最小增行/删行操作的短窗口净收益。论文中应通过 `capacity-sufficient but structurally-mismatched` 场景证明差异，否则容易被认为只是另一种跨 DC autoscaling。

### 9.3 真正可主张的创新性

建议把创新性写成“**新的耦合关系与决策准则**”，而不是“使用了新的推理引擎”：

1. **状态到结构的映射**：Prefix 覆盖率 `H_ik` 同时影响 P_i 对请求类 k 的 Prefill 工作量、有效容量 `μP_i` 和 P-D cost，而不是只作为 cache hit 统计量。
2. **结构编辑的反事实评估**：P 扩缩容被定义为关系矩阵的增行/删行/替换行；动作价值由重求解后的 `J*(S_a)` 减去启动、warmup、cache loss 等真实成本得到。
3. **存储动作的条件执行**：LMCache warmup 不是默认开启，而是与 cold 候选在同一结构位置竞争；`G_warm > G_cold` 才执行。这是存储状态参与计算重构的直接证据。
4. **增量且连续的在线闭环**：仅优化新流量，锁定 active Decode KV，限制 `|ΔP|≤1` 并设置 dwell time，使算法能落到 vLLM+LMCache，而不是停留在全局离线优化。

### 9.4 创新强度的诚实判断

这项工作的创新强度取决于实验是否证明以下三件事，而不是取决于模块数量：

- 在总 P capacity 足够时，结构收益仍能稳定发现真正的 P 位置/拓扑瓶颈；
- cache state 改变 `μP` 或 warm/cold 排序，并带来更低的重算 token、ramp-up time 或 SLO goodput，而不只是更高 hit rate；
- 相同 vLLM、LMCache、matching solver 和网络条件下，收益来自控制策略，而不是更换了基础设施。

如果上述证据成立，最稳妥的论文定位是：**面向有状态 Prefill 和跨域异构 P-D 连接的状态感知结构重构调度**。不宜宣称覆盖所有推理调度，也不宜宣称在通用吞吐上必然击败 Dynamo、Mooncake 等完整平台。

## 10 可直接用于论文的故事案例

### 10.1 场景

系统包含 Edge 和 Cloud 两个域。Edge 有 Prefill 实例 P1 与较弱 Decode 实例 D1，Cloud 有 Prefill 实例 P2 与较强 Decode 实例 D2。P1 的 vLLM Prefix Cache/LMCache 中保存热门 Prefix A，约 80% 请求使用 A。

由于命中 A，P1 的有效 Prefill 能力约为 30 req/s；P2 因没有 A，能力约为 20 req/s。当前请求率为 40 req/s，因此总容量 50 req/s 已经超过请求率。传统 autoscaler 会认为“不需要扩容”。

然而 P1/P2 到 D1/D2 的 pair cost 不同：Edge 到 D1 延迟低但 D1 较弱，Cloud 的 D2 较强但需要跨域传输 KV。固定两行结构下，部分流量只能使用高成本边，P95 TTFT 持续偏高。这是一个“容量足够但结构失配”的案例。

### 10.2 三种方法的决策

| 方法 | 看到的状态 | 决策 | 不能处理的部分 |
|---|---|---|---|
| 利用率 autoscaler | 总 μP=50 req/s，大于 λ=40 req/s | 保持 P1、P2 | 看不到高成本 P-D 边 |
| 网络感知路由 | 当前 D1/D2 的 queue、RTT、带宽 | 在已有 D 中挑较优者 | 不能新增一个合适位置的 P 行 |
| CASR | Prefix 命中、μP、D capacity、P-D cost matrix | 反事实评估 Edge cold/warm 新 P | 只做最小的结构变化 |

CASR 对同一位置生成两个候选：`+P3@Edge(cold)` 和 `+P3@Edge(warm)`。前者不加载 Prefix A，后者通过 LMCache 预热 A。对每个候选重新求解 P-D 最小成本流：

`G(a)=J*(S)-J*(S_a)-C_reconfig(a)`。

假设 `G(cold)=13`、`G(warm)=22`，则选择 `+P3@Edge(warm)`。原因不是 GPU 利用率超阈值，而是新增 P3 改变了关系矩阵、减少了跨域 KV 流量，且预热成本能被全局分配收益覆盖。

### 10.3 热点漂移的第二幕

随后请求率不变，但热点从 A 转移到 B。P1、P3 仍然缓存 A，P2 对 B 的命中率更高。传统 autoscaler 因 λ 未变而不动作；只看 cache hit 的策略可能继续保留旧状态。CASR 重新估计每个 P 的有效容量和结构价值，发现 P3 的边际贡献下降，可能停止继续复制 A、把流量转给 P2，并在满足 dwell time 后删除结构价值最低的 P 行。

### 10.4 故事到实验的映射

| 故事中的现象 | 实验构造 | 主要指标 | 要支持的论点 |
|---|---|---|---|
| 总容量足够但结构失配 | 固定 λ，增加 Edge-Cloud RTT/降低共享 WAN 带宽 | expensive-edge ratio、J、P95 TTFT、跨域 KV 流量 | capacity sufficiency 不是不重构的充分条件 |
| warm 优于 cold | 稳定 Prefix A，扫描 reuse 与 LMCache 带宽 | `Gwarm-Gcold`、ramp-up time、recomputed tokens | 存储状态直接改变计算重构决策 |
| 热点漂移 | λ 不变，A→B 切换 | μP、SLO goodput、重构次数、cache state loss | 只看请求率的 autoscaler 会误判 |
| 最小动作 | 限制 `|ΔP|≤1` 并设置 θ/dwell time | planner latency、配置翻转、GPU-hours | 方法可在线运行且不会频繁抖动 |

### 10.5 论文中的一句话版本

> 当总 Prefill 容量已经足够时，传统 autoscaler 会保持当前配置，但跨域网络与异构 Decode 能力可能使大量请求被迫走高成本 P-D 边。本文利用 vLLM/LMCache 暴露的 Prefix 状态和 KV 传输信息，将新增 P 视为关系矩阵增行，并比较 cold/warm 候选的全局反事实净收益；因此能够发现“容量不缺、结构有错”的重构机会，并只在预热成本可被系统级收益覆盖时执行存算协同。

## 11 当前实现与实验进度

当前代码已将方案中的结构评估主链路接入 LLMServingSim：

- `PrefixProfiler` 生成 block-aligned prefix class，并维护每个 P/class 的到达率、命中 token、缓存占用和衰减状态。
- `CapacityAwareFlowSolver` 支持按 P/class 命中工作量计算容量，并提供 deterministic greedy 和 OR-Tools GLOP 两种内层求解器。
- `StructuralEvaluator` 对 `keep`、`+P(cold)`、`+P(warm)` 和 `-P` 做单步反事实重求解，使用窗口收益、绝对/相对阈值和 dwell time 选择动作。
- `PrefillLifecycle` 和 `ResourceOrchestrator` 执行单步目标集合，保留 WARMING worker，支持 GPU allocation、drain、reclaim 和资源拒绝。
- 状态快照新增 `structural` 字段，记录动作、模式、基准 objective、候选 objective、收益和目标 worker。

运行结构失配实验：

```bash
cd LLMServingSim
tests/run_casr_structural_experiment.sh /tmp/casr-structural
```

该实验使用 `configs/cluster/casr_structural_mismatch.json`：生命周期按高容量配置认为只需一个 P，
但 flow solver 对当前 P 施加较小容量，候选 P 可修复 overflow。当前运行在约 108 ms 处选择一次
`+P(warm)`，objective 从 `1.012` 降至 `0.264`，窗口收益为 `0.748`，之后没有重复结构动作。

现有低/高/低 baseline 与消融仍由 `tests/run_casr_ablation.sh` 驱动；7 项 CASR 单元测试和完整消融脚本均已通过。
需要注意，这些实验仍是模拟器内的逻辑资源和本地 KV warmup，不是实际 vLLM、LMCache、Prometheus 或 Kubernetes/Ray
部署。下一阶段应补充 KV 字节/BW/RTT/queue 成本、真实 warm I/O 计时、热点漂移和 NoNetwork/NoHysteresis 对照，
再将结果用于真实系统验证。
