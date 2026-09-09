# CASR 核心算法设计

## 1. 目标与范围

CASR（Cache-Aware Split-Request）面向 Prefill/Decode 分离的 LLM serving 集群，
在请求级路由之外同时优化三件事：

1. 将具有相似前缀和负载特征的请求聚合成可观测的 prefix class。
2. 在 Prefill worker、Decode worker 以及共享链路之间做容量感知的流量分配。
3. 根据近期需求调整 Prefill worker 数量，并在 GPU 资源账本中完成申请、预热、排空和回收。

当前实现位于 `LLMServingSim/serving/casr/`，它是模拟器内的确定性控制器。资源编排接口已经抽象出来，
但尚未启动真实 Kubernetes、Ray 或 CUDA worker 进程。

## 2. 系统状态与符号

在控制周期 `t`，系统维护：

| 符号 | 含义 |
| --- | --- |
| `C` | prefix class 集合 |
| `P` | 当前可接收新请求的 Prefill worker 集合 |
| `D` | 当前可接收新请求的 Decode worker 集合 |
| `lambda_c` | class `c` 的 EWMA 到达率 |
| `h_c` | class `c` 的 EWMA 命中 token 数 |
| `r_c` | class `c` 的 EWMA 请求 token 数 |
| `f_c,p,d` | class `c` 经 Prefill `p`、交给 Decode `d` 的流量 |
| `K_p`, `K_d` | Prefill 和 Decode 的容量 |
| `B_l` | 共享链路 `l` 的容量 |

每个控制 tick 只使用 `ACTIVE` worker 参与求解。`WARMING`、`DRAINING` 和 `INACTIVE`
worker 不会被新计划选中。

## 3. Prefix class 构造

### 3.1 隐私保护的前缀标识

对请求输入 token 取前缀，最多保留 512 个 token，然后按 KV block size 对齐。
对对齐后的 token 序列计算 BLAKE2 摘要，得到 `prefix_id`。快照只保存摘要，不保存原始 token，
因此控制器可以统计复用而不会把请求内容写入实验结果。

class 标识由以下字段共同构成：

```text
class_id = model + prefix_id + input_bucket + output_bucket
```

这样，相同前缀但模型或输出规模不同的请求不会被错误地合并。状态按
`(prefill_instance_id, class_id)` 维护，因此可以观察某个 prefix 在不同 Prefill worker 上的局部命中情况。

### 3.2 EWMA 观测

每个状态记录：

- `arrival_rate_ewma`：近期到达率，用于需求估计；
- `reuse_ewma`：prefix 重用比例；
- `hit_tokens_ewma`：命中的 token 数；
- `npu_hit_tokens` 和 `storage_hit_tokens`：命中层级诊断；
- `requested_tokens`：请求 token 总量。

没有新请求时，到达率按配置的半衰期衰减。这一步使高峰结束后需求能够自然下降，避免扩容后的 worker 永久保持 ACTIVE。
求解时使用有效 Prefill 工作比例：

```text
hit_c = min(0.95, h_c / max(1, r_c))
work_c = max(0.05, 1 - hit_c)
```

`0.95` 和 `0.05` 防止完全命中时容量约束失去可见性，也避免新 class 被估计为零工作量。

## 4. 流量分配模型

### 4.1 流守恒

对每个 prefix class，所有 P/D pair 的流量必须覆盖该 class 的需求：

```text
sum(p in P, d in D) f_c,p,d = lambda_c
f_c,p,d >= 0
```

Prefill 负载按未命中工作比例折算，Decode 负载按完整请求流量计算：

```text
load_p = sum(c,d) f_c,p,d * work_c
load_d = sum(c,p) f_c,p,d
```

共享链路 `l` 上的负载是所有经过该链路的 pair 之和：

```text
load_l = sum(c,p,d: l carries (p,d)) f_c,p,d
```

共享链路可以通过 `pairs` 限定经过它的 P/D pair；未指定 pair 时表示所有 pair 共用该链路。

### 4.2 Greedy baseline

Greedy 是无外部依赖的确定性基线。它按照 `lambda_c` 降序处理 class，并为每个 class 枚举所有 P/D pair。
候选 pair 的代价为：

```text
cost(p,d) = load_p / K_p + load_d / K_d + distance(p,d)
            + penalty * (overflow_p + overflow_d + overflow_l)
```

其中：

- `distance(p,d)` 是 P/D 的 NPU 起始位置差，当前按 `0.001` 比例进入代价；
- `overflow_*` 是加入当前流量后的容量超额；
- `penalty` 默认通过 `overflow_penalty` 配置。

选择最小代价 pair 后立即更新 P、D 和共享链路负载，再处理下一个 class。代价排序包含 worker ID，
因此相同输入会产生相同计划。

伪代码如下：

```text
for c in sort(classes, descending=lambda_c):
    flow = max(lambda_c, 1)
    best = None
    for p in P:
        for d in D:
            evaluate utilization, topology distance, and three overflows
            keep lexicographically smallest (cost, p.id, d.id)
    assign(c, best.p, best.d, flow)
    update p_load, d_load, and link_load
```

Greedy 的优点是低开销、可复现、无需 OR-Tools；缺点是先分配的 class 会影响后续 class，不能保证全局最优。

### 4.3 连续 LP

当配置 `solver: lp` 且安装 OR-Tools 时，使用 GLOP 求解连续流。引入三类松弛变量：
`s_p`、`s_d`、`s_l`，分别表示 Prefill、Decode 和共享链路超载：

```text
min  sum(c,p,d) distance(p,d) * f_c,p,d
   + penalty * (sum(p) s_p + sum(d) s_d + sum(l) s_l)
```

约束为：

```text
sum(p,d) f_c,p,d = lambda_c                         for every c
sum(c,d) f_c,p,d * work_c <= K_p + s_p               for every p
sum(c,p) f_c,p,d <= K_d + s_d                       for every d
sum(c,p,d: l carries (p,d)) f_c,p,d <= B_l + s_l    for every l
f_c,p,d, s_p, s_d, s_l >= 0
```

LP 会先按 `class_id` 聚合同一 class 在多个 P 上的观测，避免重复观测被后一个 row 覆盖。
输出除了 flow，还记录 objective 和三类 overflow，便于实验审计。若 OR-Tools 不可用，自动回退到 greedy，
并在 solver 诊断中写入 `fallback: ortools unavailable`。

当前低/高/低 workload 中 LP 和 greedy 结果相同。这表示该 workload 没有触发全局精确分流的额外收益，
不能据此宣称 LP 始终优于 greedy；应在容量临界、多个共享链路竞争的 workload 上单独比较。

## 5. 从 flow 到在线路由

求解器输出的 `f_c,p,d` 通过两级权重转为 `AffinityPlan`：

```text
P_weight(c,p) = sum_d f_c,p,d / sum_p,d f_c,p,d
D_weight(p,c,d) = f_c,p,d / sum_d f_c,p,d
```

控制器为计划生成单调递增的 `version` 和 `expires_at_ns`。计划过期或没有匹配 class 时，路由器回退到普通路由，
避免旧计划无限期影响新流量。

Prefill 侧使用 deterministic deficit routing：对每个候选 P 计算
`desired - observed`，优先选择当前欠分配的 worker。该策略比随机权重更容易复现实验，同时仍能逼近目标比例。

Decode 侧按 `(prefill_id, class_id)` 选择目标 D。每个 Prefill batch 只绑定一个 Decode target，
从而满足 ASTRA-Sim receiver graph 对单个 batch 的拓扑约束；不同 batch 可以根据计划分散到不同 Decode worker。

## 6. 控制器时序

每个控制 tick 的处理顺序如下：

```text
请求事件
   |
   v
PrefixProfiler.snapshot(t)
   |
   v
PrefillLifecycle.update(t)
   |  计算目标 worker 数，执行资源 acquire/reuse/reject/reclaim
   v
筛选 ACTIVE P/D
   |
   v
Policy.solve(snapshot, P, D, solver)
   |
   v
校验 class 流守恒、flow 正值、worker ID
   |
   v
生成 AffinityPlan + prefix warmup
   |
   v
Router 安装带 TTL 的计划
```

生命周期先于求解，保证新计划只引用当前可接收请求的 worker。目标 Prefill 数由需求和
`prefill_capacity` 决定，并受 `min_active_prefill`、`max_active_prefill` 限制。

## 7. 资源级扩缩容

启用 `casr.resources` 后，`ResourceOrchestrator` 为每个 worker 建立 GPU 资源所有权：

1. **Scale-out**：在节点上逐卡匹配 `num_npus` 和每卡显存需求；成功后记录 GPU ID，worker 进入 `WARMING`。
2. **Ready**：经过 `startup_ms` 后进入 `ACTIVE`，才允许接收新请求。
3. **Scale-in**：先进入 `DRAINING`；没有 running/waiting 请求后进入 `INACTIVE`，再等待 `reclaim_ms`。
4. **Release**：回收 allocation，更新节点剩余 GPU/显存；回收窗口内重新需要该 worker 时可以复用 allocation。
5. **Reject**：节点不存在或 GPU/显存不足时不伪造扩容，记录 `resource_reject` 事件。

每个 tick 的资源快照包括 allocation、节点剩余资源、pending reclaim 以及 acquire/release/reject 事件。
这使 GPU-seconds、启动开销和资源拒绝都可以从结果中审计，而不只是观察 admission flag。

## 8. 实验解释

当前三阶段 trace 为低负载 → 高负载 → 低负载，固定模型、输入输出长度和 prefix 复用，仅改变到达率。
基线和消融结果如下：

| Case | 说明 | 平均 latency | 高峰 P95 | GPU-seconds |
| --- | --- | ---: | ---: | ---: |
| `fixed2` | 固定 2 个 Prefill | 929.179 ms | 1210.446 ms | 20.294 |
| `fixed1` | 固定 1 个 Prefill | 929.346 ms | 1208.582 ms | 15.265 |
| `dynamic_lp` | 完整 CASR，LP | 785.159 ms | 804.702 ms | 16.904 |
| `dynamic_greedy` | 完整 CASR，greedy | 785.159 ms | 804.702 ms | 16.904 |
| `routing_only` | 保留路由，去掉资源级扩缩容 | 785.159 ms | 804.702 ms | 18.947 |
| `no_prefix` | 去掉 prefix 复用信息 | 924.704 ms | 1213.234 ms | 14.417 |

在这条 trace 上，完整 CASR 相比固定 2P 的平均 latency、峰值 P95 和 GPU-seconds 分别约下降 15.5%、33.5% 和 16.7%。
`routing_only` 延迟接近完整 CASR 但 GPU-seconds 更高，说明异构路由主要贡献性能，资源编排主要贡献容量释放。
`no_prefix` 无法形成稳定的历史需求状态，峰值延迟回到约 1.21 s，体现 prefix 观测对预测和扩缩容的作用。

实验必须同时报告：控制 tick 间隔、`startup_ms`、`reclaim_ms`、worker 配置、solver backend、overflow 诊断和资源事件。
尤其要把 scale-out 的启动成本单列，不能将动态扩缩容收益解释成没有代价的即时扩容。

## 9. 已实现能力与边界

当前已实现：

- prefix 摘要、分桶、命中统计和半衰期衰减；
- capacity-aware greedy 与可选 OR-Tools GLOP LP；
- P/D/link 三层容量约束和 overflow 审计；
- 带 version/TTL 的亲和计划与确定性在线路由；
- Prefill batch 到 Decode 的单目标 handoff；
- 节点级逐卡 GPU/显存账本、WARMING/DRAINING/资源回收；
- baseline、routing 和 prefix 消融实验。

当前边界：

- worker 对象仍由一次 ASTRA-Sim 仿真预先创建，资源编排不会动态改变静态网络拓扑；
- 尚未接入 Kubernetes、Ray、容器启动或真实 CUDA 上下文迁移；
- flow 是连续值，在线路由以权重近似，尚未实现请求级整数规划；
- 当前成本函数主要包含容量、拓扑距离和超载惩罚，尚未纳入真实网络带宽、功耗、价格或 SLA 多目标；
- prefix warmup 在模拟器中写入 KV cache，用于模拟预热收益，不代表跨进程 KV 迁移协议。

## 10. 后续验证重点

1. 构造共享链路饱和和 P/D 容量临界 workload，验证 LP 相对 greedy 的差异。
2. 扫描控制 tick、`startup_ms`、`reclaim_ms` 和 overflow penalty，量化控制滞后与抖动。
3. 增加 prefix 复用率、prefix 长度和输入输出长度的正交实验。
4. 将资源事件、计划版本、路由比例和请求 latency 统一关联，检查扩缩容期间是否出现过期计划或错误 receiver。
5. 设计真实编排后端适配层，再评估 Kubernetes/Ray 的启动、失败重试和 GPU 抢占语义。
