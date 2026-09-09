# LLMServingSim 面向 CASR 的增强需求

## 1. 结论

LLMServingSim 已经适合作为 CASR 的执行与性能仿真底座：它已有连续 batching、P/D 分离、KV block cache、CPU/CXL 下层缓存、异构实例 profile、ASTRA-Sim 网络后端和逐请求延迟结果。

但它当前模拟的是“**固定实例集合 + RR/LOAD 请求路由 + 固定拓扑下的 P/D handoff**”。CASR 所需的是“**状态可观测的 P-D 关系矩阵 + 可下发 affinity + 可编辑 P 行 + 可计量 warm/cold 成本**”。因此应补齐控制面与状态模型，避免修改 attention、KV block 基础算法或 ASTRA-Sim 内核。

增强优先级：先完成 P0 才能验证 CASR 主张；P1 用于提升跨域和存储结论可信度；P2 是后续扩展，不进入第一篇论文主线。

## 2. 当前能力与缺口

| 维度 | LLMServingSim 当前能力 | CASR 缺口 | 优先级 |
|---|---|---|---|
| P/D 执行 | `pd_type: prefill/decode`，模拟 Prefill KV 传给 Decode | 多 P 到多 D 的显式 pair affinity 与流量比例 | P0 |
| 快层路由 | RR、随机、LOAD；P 完成后再选 D | 控制器下发的 class-aware `π_ijk`、临时降级规则 | P0 |
| Prefix Cache | token IDs → chained hash → block 命中；NPU/CPU/CXL 层 | prefix class、热度、每 P 的 `H_ik` 与缓存价值 | P0 |
| 输出 | TTFT/TPOT、排队、累计 hit ratio、pool 使用率 | 每请求命中、P/D pair、KV bytes、每周期状态/动作日志 | P0 |
| 网络 | ASTRA-Sim 分析型后端、cluster link 参数 | 域/链路级可见状态、P-D pair 专属 RTT/BW、共享 WAN 竞争 | P0 |
| 弹性 | 固定实例集合 | `ACTIVE/WARMING/DRAINING/INACTIVE` P 生命周期与启停成本 | P0 |
| 存储 | 本地 NPU + 节点 CPU pool 或全局 CXL pool | 跨域远端 KV、副本可达性、显式 warmup 流 | P1 |
| 容量 | 硬件 profile、每实例 scheduler 限制 | 对 class 的 cache-aware `μP_i,k` 和 residual `μD_j` 导出 | P0 |
| 工作负载 | token-ID JSONL、到达时间、agentic session | 可控热点稳定/漂移、prefix 重用率、prefix class 标签 | P0 |
| 评估 | 固定配置、确定性回放 | 动作日志、反事实结果、基线策略和重复实验编排 | P1 |

## 3. P0：必须实现的最小控制面

### 3.1 Prefix class 与热度状态

#### 目标

将现有“单个 request 是否命中多少 token”的内部状态，转换成控制器可消费的统计量：`H_ik`（P_i 对 class k 的平均命中长度）、`r_k`（热度）和 `B_ik`（驻留缓存量）。

#### 新增数据模型

```text
PrefixClass
  class_id: str
  model_id: str
  prefix_id: str
  input_bucket, output_bucket
  arrival_rate_ewma: float
  reuse_ewma: float
  mean_input_tokens, mean_output_tokens
  kv_bytes_per_request

PrefixState[p_instance][class_id]
  hit_tokens_ewma: float        # H_ik
  request_count_window: int
  cached_blocks: int
  cached_bytes: int
  last_access_ns: int
  evicted_blocks_window: int
```

`prefix_id` 应由 `model_id + prefix_length + last_hash_of_prefix` 派生；只使用完整 KV block，避免泄露原始 prompt。为避免一个很长 prompt 的尾部差异使所有请求分裂成独立类，prefix 长度可截断为控制的候选层级，例如 128、256、512 token。

#### 事件 hook

不改变 `TieredKVCacheManager` 的分配和 eviction 语义，只添加 observer：

| 事件 | 现有位置 | 新增记录 |
|---|---|---|
| 请求到达 | `Router._load_*` / `route_arrived_requests` | class、arrival、P 选择 |
| cache lookup | `get_computed_blocks` | NPU/storage hit token、prefix id |
| cache insert | `cache_full_blocks` | 新增 block/字节、P 的驻留状态 |
| eviction | `BlockPool` 从 free list 取走带 hash 的 block | 被驱逐 prefix、字节、时间 |
| P/D handoff | `transfer_prefill_request` | 源 P、目标 D、class、KV bytes |
| 请求完成 | `Scheduler.add_done` | 实际 TTFT/TPOT、重算 token |

#### 输出

新增两类输出，格式优先 JSONL，便于离线复算：

```text
runs/<id>/requests.jsonl       # 每个完成请求一条
runs/<id>/control_ticks.jsonl  # 每个 T_ctrl 一条状态、候选与动作
```

现有 CSV 保留兼容；扩展字段至少包含：`class_id,prefix_id,p_instance,d_instance,npu_hit_tokens,storage_hit_tokens,recomputed_tokens,pd_kv_bytes,affinity_version`。

### 3.2 CASR affinity router

#### 目标

让慢层求出的 `f_ijk` 真正作用于仿真，而非只作为离线分析结果。

#### 接口

```python
AffinityPlan(
    version: int,
    expires_at_ns: int,
    prefill_weights: dict[class_id, dict[p_id, float]],
    decode_weights: dict[tuple[p_id, class_id], dict[d_id, float]],
    fallback_decode_ids: dict[tuple[p_id, class_id], list[d_id]],
)
```

路由规则：

1. 请求到达时按 `prefill_weights[class]` 选可 admission 的 P；
2. P 完成后只在 `decode_weights[(P,class)]` 的 D 集合中选择最小实时边际代价的 D；
3. 该集合暂时无容量时，依序尝试 `fallback_decode_ids`；
4. 都不可用时保留在等待队列或计入 overflow，不能悄悄绕过 affinity；
5. 运行中的 Decode 请求不因 plan 更新而迁移。

第一版可将连续流量比例随机化为 weighted rendezvous/weighted round-robin，保证长窗口流量收敛到 LP 解；不需要实现新的在线最优 router。

### 3.3 Controller tick、状态机与 P 行编辑

#### 控制周期

在模拟器主循环中按模拟时间触发，不按宿主机墙钟触发：

```text
if current_sim_time >= next_control_tick:
    snapshot = collector.snapshot()
    plan, action = casr_controller.decide(snapshot)
    executor.apply(action)
    router.install(plan)
```

#### P worker 生命周期

```text
INACTIVE → WARMING → ACTIVE → DRAINING → INACTIVE
              │         │
              └─────────┘ (warm 失败/超时则 cold ACTIVE)
```

| 状态 | 是否接新请求 | 现有请求 | 容量计算 |
|---|---:|---|---|
| `INACTIVE` | 否 | 无 | 0 |
| `WARMING` | 否 | 无 | 0，记录 warm I/O |
| `ACTIVE` | 是 | 正常执行 | 进入 `μP_i,k` |
| `DRAINING` | 否 | 完成后退出 | 不再作为新流量供给 |

第一版不真实创建/销毁 Python Scheduler。每个候选 P 预先存在，只在 router admission 中启用或禁用；启动延迟、warmup 和 drain 用事件及资源占用建模。这保证第一版变量只来自 CASR 策略，不来自容器编排。

### 3.4 多类最小成本流和容量导出

新增独立 `serving/casr/` 包，不嵌入 Scheduler：

```text
state.py            Snapshot / PrefixClass / WorkerState
capacity.py         μP_i,k, residual μD_j
cost.py             A_ij, c_ijk, link constraints
flow.py             OR-Tools LP 或 min-cost flow wrapper
evaluator.py        候选、warm 集合、G_W
controller.py       tick / dwell time / action selection
```

模拟器应暴露以下只读输入：

- P 的 waiting/running、实际计算 token、命中/重算 token、可用 block；
- D 的 active request、剩余 output token、队列和 profile；
- P-D KV token/bytes、RTT/BW、链路占用；
- prefix class 的 arrival/reuse EWMA。

内层 solver 使用 `f_ijk`，而非仅 `f_ij`；必须加入 overflow sink 和链路带宽约束，确保任何负载下都能返回可比较的反事实结果。

### 3.5 域、pair 和共享链路建模

当前配置可表达 node、实例和全局 link，但 CASR 需要显式表达域与 pair/link 状态。建议向 cluster config 增加逻辑层，不修改 ASTRA-Sim 原生网络 JSON：

```json
{
  "casr": {
    "domains": ["edge-a", "edge-b", "cloud"],
    "instances": {
      "0": {"domain": "edge-a", "role": "prefill", "pool": "edge-a-p"},
      "1": {"domain": "cloud", "role": "decode"}
    },
    "links": [
      {"id": "wan-a-cloud", "src": "edge-a", "dst": "cloud",
       "rtt_ns": 20000000, "bw_gbps": 20, "shared": true}
    ]
  }
}
```

要求：

- 从 `(P_i,D_j)` 确定可行性、RTT、带宽和共享链路 ID；
- 多条 P-D flow 使用同一 WAN 时要共同消耗预算；
- 支持在控制 tick 修改 link profile（带宽下降、RTT 上升）；
- 对每个 P-D handoff 记录实际/预计 KV bytes；
- 逻辑域可先映射到同一物理 node，用限速和延迟复现跨域；真实多节点为后续校准。

## 4. P1：增强存储与实验可信度

### 4.1 显式 warmup 和远端 Prefix 副本

CPU/CXL pool 已能模拟下层 KV recall，但不能表达“控制器选择把某些 prefix 预取到新 P”。应加入：

- `WarmRequest(p_id, prefix_ids, source_pool, byte_budget)` 事件；
- 以 block/chunk 为单位的存储读取、网络传输和 NPU cache insert；
- warm 与 P-D KV transfer 共用链路带宽时的竞争；
- 预热中断、超时、缓存空间不足、部分 warm 成功；
- `warm_bytes,warm_duration_ns,warm_hit_tokens_after_ready` 指标。

新 P 的初始 `H_new,k` 只能由实际成功加载的 block 决定，不能在决策后立即假定为满命中。

### 4.2 Scale-in 的 cache-loss 成本

scale-in 不应只删除 P 的吞吐。模拟器应在 drain 完成时输出该 P 上被删除的 prefix class、块数和预计未来重算 token；控制器用它组成 `C_cache_loss(a)`。

第一版可以用近期 `reuse_ewma × cached_bytes` 的估计；P1 再通过“影子保留该行”的反事实重放验证估计误差。

### 4.3 Profile 与容量校准

对于每种 GPU、模型、P/D role、输入/输出 bucket，应导出可供 CASR 查询的 profile：

```text
prefill_time(gpu, effective_input_tokens, batch_state)
decode_time(gpu, output/context, batch_state)
kv_bytes(model, tokens, dtype)
startup_time(domain, gpu_type)
warm_load_bandwidth(source, destination)
```

LLMServingSim 的 layerwise profile 已可覆盖前 3 项的一部分。需要补充的是 role-specific 聚合查询、启动/预热测量参数和不确定性区间。

### 4.4 工作负载生成器

新增合成 trace 生成器，严格提供 `input_tok_ids`，否则 Prefix Cache 永远无法命中。需可独立扫描：

- stable hot prefix、Zipf 多热点、周期热点；
- 热点 A→B 漂移而到达率保持不变；
- prefix 复用率、长度、输入/输出长度；
- edge/cloud 到达比例；
- burst、链路退化、D 异构度；
- 随机种子和已知 ground-truth class 标签。

这比直接使用只含长度的 trace 更重要，因为 CASR 的关键证明是“相同 λ、不同 Prefix 状态”的行为差异。

### 4.5 可复现的实验记录

每个 run 应落盘：

```text
config.resolved.json          # 所有默认值展开后的配置
profiles.lock.json            # profile bundle/version/hash
events.jsonl                  # request/cache/warm/link/lifecycle 事件
control_ticks.jsonl           # S_t、候选、J、G_W、最终动作
metrics.parquet or csv        # request/instance/link/prefix 多维指标
seed.txt
```

这使同一 trace 下的 Static、LoadScale、NetworkHeuristic、SG-NoState、CASR 只差控制策略，避免基础配置不一致。

## 5. P2：明确延后

以下功能很有研究价值，但不应阻塞 CASR 第一版：

- D 行的增删、D placement 和 P/D 双边编辑；
- active Decode KV migration；
- 多副本全局 cache placement 与复杂 eviction；
- 真实 Kubernetes/Ray worker 创建、故障恢复和服务发现；
- NS-3 级逐包网络仿真；
- 学习型预测器、强化学习和长期多步规划；
- KV 压缩、量化和模型权重跨域迁移。

分析型网络后端加“共享带宽预算 + 控制 tick 参数变更”已经足以验证第一版；不必等待 NS-3。

## 6. 实施顺序和验收

| 阶段 | 改动 | 验收条件 |
|---|---|---|
| A：观测 | PrefixProfiler、请求/事件输出 | 能按 P 与 prefix class 画热度、命中长度、缓存量 |
| B：固定结构 | `f_ijk` solver、AffinityPlan、CASR router | 同一 P/D 集合下比 LOAD/nearest-D 更低 `J` 或 P95 TTFT |
| C：多域 | CASR logical domain/link 配置、共享 WAN 约束 | WAN 退化时 pair flow 和链路流量按预期变化 |
| D：结构编辑 | P 状态机、cold、scale-in、`G_W`、dwell time | 总 P 容量足够但拓扑失配时仍能有正收益动作 |
| E：存算协同 | warm request、远端 prefix、副本/带宽竞争 | 复用率或 BW 改变时 warm/cold 动作排序翻转 |
| F：校准 | 小规模 vLLM/LMCache profile/trace 对照 | 仿真趋势、warm 时长和命中行为方向一致 |

每完成一个阶段，保留一个 10--100 请求的 deterministic smoke case 和一个饱和/热点漂移回归 case；不要只依赖当前的无命中轻负载示例。

## 7. 推荐的第一批改动

建议从下面四项开始，代码小且能立刻形成 CASR 的可观测闭环：

1. 为 Request 输出补充 `class_id,prefix_id,P,D,npu_hit_tokens,storage_hit_tokens,pd_kv_bytes`；
2. 添加 `PrefixProfiler`，产生每控制周期的 `PrefixState[P][class]` 快照；
3. 将 Router 的 P/D 选择改成可安装的 `AffinityPlan`，先用静态手写计划验证；
4. 添加两域、多 P、多 D 的逻辑 cluster config 和带共同 prefix token IDs 的 synthetic trace。

这四项完成后，再接 OR-Tools 的 `f_ijk` LP。届时能够把“算法输出一个矩阵”变成“矩阵改变仿真请求路径和测量结果”，是进入 Structural Gain 的可靠起点。

## 8. 第一轮实现状态（2026-09-07）

第一轮已完成可运行的 P0 控制面基线，代码位于 `LLMServingSim/serving/casr/`。

| 能力 | 状态 | 说明 |
|---|---|---|
| `class_id` / 隐私保护 `prefix_id` | 已实现 | 由 block 对齐的 token ID 前缀生成 BLAKE2 摘要；无 token ID 的请求标为 `none`。 |
| P×class 热度、复用与命中观测 | 已实现 | `PrefixProfiler` 维护到达率 EWMA、复用、NPU/存储命中 token、队列和缓存容量。 |
| 控制 tick 状态输出 | 已实现 | `--casr-state-output` 写 JSONL；建议控制周期不小于 100 ms，避免大量小文件写入扭曲运行时间。 |
| 版本化亲和计划 | 已实现 | `AffinityPlan` 可原子替换；Router 对新到达请求使用 P/D 亲和权重和 fallback。 |
| 基线容量感知计划 | 已实现 | `CASRController` 是无外部依赖、确定性的贪心基线；按 P/D 队列负载与 prefix hit 估计分配 class。不是最终 LP。 |
| 请求级指标 | 已实现 | CSV 新增 class、prefix、P/D、两级 hit token、P/D KV bytes、plan version。 |
| 非配对 P→D 图传输 | 已实现（单目标 P batch） | Batch 将 Decode 的全局 NPU 起点写入 trace header，Chakra 据此生成发送/接收 ET 图，不再固定 `npu + num_npus`。 |

运行示例：

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_pd_instance.json \
  --dataset workloads/example_trace.jsonl --num-reqs 10 \
  --enable-casr --casr-control-interval-ms 100 \
  --casr-state-output outputs/casr_state_{run_id}.jsonl \
  --output outputs/casr_requests_{run_id}.csv
```

已用 `single_node_pd_instance.json` 和 `example_trace.jsonl` 的一请求 P/D case 冒烟：请求完成，输出 CSV 含新增列，最终 JSONL snapshot 含 prefix class、缓存和实例状态。

### 当前边界

每张 trace 仍只表达一个 Decode receiver，但 scheduler 已按目标分桶并逐 batch 执行；因此不同 class 可以在同一个 P worker 上进入不同 D，只是不会混入同一 batch。P/D 的 TP NPU 数目前必须相等，避免 KV rank 映射歧义。

尚未实现：逻辑 domain/shared-link 配置、显式远端 warmup 请求与 cache loss。P 生命周期及资源级扩缩容已在下一轮实现；仍未接入真实 Kubernetes/Ray 进程编排。

## 9. 第二轮实现状态：可执行 f_ijk 与动态 P→D（2026-09-07）

第二轮完成了从控制面矩阵到 ASTRA-Sim 图执行的最小闭环：

- `serving/casr/flow_solver.py` 提供确定性的 `CapacityAwareFlowSolver`，输出显式 `FlowAssignment(class_id, prefill_id, decode_id, flow, cost)`；成本含 P/D 已分配容量、prefix hit 降低的 P 工作量和拓扑距离代理项。
- `Router` 对亲和权重使用确定性 deficit routing，而非始终取最大权重，因而可执行未来 LP 返回的分数 flow。
- Prefill scheduler 每次只从一个 Decode 目标取请求组成 batch；其它目标的请求保留在队列，下一 batch 再运行，避免一张 Chakra 图包含多组不可表达的 receiver。
- 非历史配对的 P→D 使用目标 Decode NPU 的 receiver ET 图。主循环在该 NPU 下次轮询时显式交付 receiver workload，并将 receiver 完成回报到源 Prefill batch；完成后才把 Request 放入目标 Decode scheduler。
- `--casr-state-output` 的每个 snapshot 新增 `flows`，用于画出实际求解的 `f[class,P,D]`。

验证：使用 `single_node_pd_instance.json`、`example_trace.jsonl`、两个请求、100ms control tick 的回归已完成；最终状态显示两个 request 都退出，并保存两行 class 的 flow 矩阵。该用例的目标 D 仍只有一个，验证的是动态 receiver 交付与完成依赖；多 P、多 D 配置将用于下一轮验证容量和分流效果。

补充回归：`configs/cluster/casr_two_prefill_two_decode.json` 已提供 2P×2D 最小拓扑。以 `example_trace.jsonl` 的前三个请求运行后，三个请求全部退出，最终 `flows` 同时包含 Decode 2 和 Decode 3，证明 flow 分配、按目标分桶、动态 receiver 交付和 Decode handoff 在多目标下连通。

## 10. 共享容量与 overflow 配置

cluster JSON 可选增加 `casr` 段。容量单位是控制窗口中的 flow 单位（当前 baseline 以请求到达率估计），因此应与同一运行的 `arrival_rate_ewma` 标度一起校准：

```json
"casr": {
  "prefill_capacity": {"0": 20, "1": 20},
  "decode_capacity": {"2": 40, "3": 40},
  "overflow_penalty": 10,
  "shared_links": [
    {"id": "wan-a", "capacity": 15, "pairs": [[0, 2], [1, 2]]}
  ]
}
```

`pairs` 列出经过该共享资源的 `(prefill_instance_id, decode_instance_id)`；省略或为空表示所有 P→D pair 共用该链路。每个 flow 输出 `link_ids`、`prefill_overflow`、`decode_overflow`、`link_overflow` 和总 `cost`。这给 LP 替换层提供了与当前贪心 baseline 相同的约束和审计面。

`casr.lifecycle` 可模拟预创建 Prefill worker 的 admission 生命周期：

```json
"lifecycle": {
  "min_active_prefill": 1,
  "max_active_prefill": 4,
  "warmup_ms": 500
}
```

控制器根据 flow demand 与 `prefill_capacity` 选择目标 ACTIVE 数。空闲 worker 可 `deactivate`；有在途请求的 worker 先进入 `DRAINING`，排空后才 `INACTIVE`；重新启用会经过 `WARMING`。这只模拟 admission/预热时间和容量，不创建或销毁真实进程。每个 tick 的 `lifecycle` 字段记录动作。

控制器还会对 flow 指向 P 的已知 class 执行 prefix warmup：以代表性 token 序列写入完整的未固定 NPU KV block，后续请求照常经过 cache lookup、命中和 eviction。`warmups` 记录写入字节数；token IDs 仅驻留进程内用于构造 cache hash，绝不写入 snapshot。

### 10.1 资源级编排

当 `casr.resources` 存在时，生命周期动作由 `ResourceOrchestrator` 执行节点级资源账本，而不只是切换 admission flag。每个 worker 必须获得满足其 `num_npus` 和逐卡显存需求的 GPU allocation；资源不足时 scale-out 被拒绝并记录原因。scale-in 先 drain，随后经过 `reclaim_ms` 才释放 GPU；scale-out 获得资源后经过 `startup_ms` 进入 `WARMING`，完成后才能接收新请求。

资源快照记录每个 worker 的 `gpu_ids`、显存占用、节点剩余 GPU/显存、pending reclaim 和 acquire/release/reject 事件。当前实现是模拟器内置的确定性本地编排后端，真实进程创建仍属于部署层，不会在静态 ASTRA-Sim 拓扑中伪造动态 CUDA worker。

### 10.2 三阶段弹性实验

已增加 `workloads/casr_elasticity_low_high_low.jsonl` 和
`tests/run_casr_elasticity_comparison.sh`。实验固定模型和 prefix 复用，
只改变到达率：前 1 秒为 5 req/s，中间 2 秒为 50 req/s，最后 1 秒恢复
为 5 req/s。三组分别是固定 2P、固定 1P 和动态 CASR；动态组使用
`startup_ms=250`、`reclaim_ms=50`，记录每个控制 tick 的 GPU allocation、
WARMING、drain 和 release。

当前运行结果：动态 CASR 在高峰期从 1P 扩展到 2P，高峰结束后回收一个
Prefill GPU；相对于固定 2P，GPU-seconds 从约 20.29 降至 16.90，平均延迟
从约 940.16 ms 降至 790.57 ms。启动成本为一次 250 ms，应在论文结果中
单独报告，不能并入静态吞吐收益。

### 10.3 基线与消融

`tests/run_casr_ablation.sh` 在同一条低/高/低 trace 上运行以下矩阵：

| Case | 去掉的能力 | 平均 latency | 高峰 P95 | GPU-seconds |
| --- | --- | ---: | ---: | ---: |
| fixed2 | CASR，固定 2P | 929.179 ms | 1210.446 ms | 20.294 |
| fixed1 | CASR，固定 1P | 929.346 ms | 1208.582 ms | 15.265 |
| dynamic_lp | 无 | 785.159 ms | 804.702 ms | 16.904 |
| dynamic_greedy | LP，仅保留 greedy | 785.159 ms | 804.702 ms | 16.904 |
| routing_only | 资源级扩缩容 | 785.159 ms | 804.702 ms | 18.947 |
| no_prefix | prefix 复用信息 | 924.704 ms | 1213.234 ms | 14.417 |

完整 CASR LP 相对固定 2P 将平均 latency 降低约 15.5%，高峰 P95 降低
约 33.5%，同时 GPU-seconds 降低约 16.7%；动态资源动作包含一次
250 ms scale-out 和两次 resource release。`routing_only` 与完整 CASR 延迟
接近但资源消耗更高，说明异构路由贡献了主要性能收益，资源编排贡献了
容量释放收益。`no_prefix` 的 111 个请求各自形成独立 prefix class，控制器
无法从历史复用中估计稳定需求，因而高峰 P95 回到约 1.21 s；这是 prefix
状态对弹性决策作用的消融证据。LP 与 greedy 在这条 workload 上结果相同，
说明该负载没有触发精确 LP 相比贪心的额外分流收益，后续应增加容量临界和
共享链路竞争场景单独评估 solver 差异。

## 11. 精确 LP 后端

`requirements-casr.txt` 提供可选 OR-Tools GLOP 后端。安装后，在 `casr` 段设置 `"solver": "lp"` 即可求解连续变量 `f[class, P, D]`：

```bash
pip install -r requirements-casr.txt
```

LP 约束包括每个 class 的流守恒、按 prefix hit 折算的 P 容量、D 容量和每个 shared link 的容量；P/D/link overflow 均为带罚分的 slack。`solver` snapshot 会记录 backend、objective 以及三个层面的 slack。若 OR-Tools 未安装，配置不会阻断仿真，而是回退到确定性 greedy 并记录 `fallback: "ortools unavailable"`。

已在 2P×2D、三请求、共享链路和容量约束的拓扑上以 `ortools-glop` 完成端到端回归；请求全部退出，LP objective 已写入最终 snapshot。

此外，既有服务层回归的 `multi` 与 `pd` 场景在本轮改动后仍与基线 clock 完全一致。

## 12. 接入研究算法的策略接口

不需要修改事件循环即可接入研究算法：传入 `--casr-policy package.module:Class`。策略类接收 cluster 的 `casr` 配置，并实现：

```python
from serving.casr import FlowAssignment

class MyPolicy:
    def __init__(self, config):
        self.config = config

    def solve(self, snapshot, prefill, decode, solver):
        # snapshot: prefix_states、实例缓存/队列/生命周期状态
        # prefill/decode: 当前 ACTIVE worker
        # solver: 可选择复用内置 greedy/LP 作为子问题求解器
        return [
            FlowAssignment(class_id, p_id, d_id, flow, cost)
        ]
```

框架在安装计划前校验 class 流守恒、正 flow 和 ACTIVE P/D ID；随后自动将 flow 转成分数亲和权重、按 Decode 目标分 batch、执行动态 KV receiver 图、warmup、生命周期和输出指标。`builtin` 是默认策略，使用 `casr.solver` 的 `greedy` 或 `lp`。

可执行的最小参考实现位于 `serving/casr/example_policy.py`；已通过 `--casr-policy serving.casr.example_policy:RoundRobinPolicy` 在 2P×2D、三请求配置上完成端到端回归。
