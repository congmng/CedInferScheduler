# CASR 存算协同调度测试报告

## 1. 测试结论

本次测试针对当前 `LLMServingSim` 中的 CASR 调度链路进行验证。结果表明：

- 自动化回归测试 `21/21` 通过，覆盖 Prefix class、缓存容量、流量守恒、网络代价、遥测、资源生命周期、结构反事实和 workload 生成器。
- 在固定低/高/低负载 trace 上，完整 CASR 的平均 latency 为 `785.159 ms`，峰值阶段 P95 为 `804.702 ms`。
- 相比固定结构 baseline，完整 CASR 平均 latency 降低约 `15.5%`，P95 降低约 `33.5%`。
- 去掉网络代价后，平均 latency 上升到 `839.796 ms`，P95 上升到 `1126.634 ms`，说明 RTT、带宽和队列代价会影响路由结果。
- 去掉迟滞机制后触发 `4` 次 `+P`，完整 CASR 未重复执行结构动作，验证了收益阈值和 dwell time 的抑制作用。
- 结构失配场景在约 `108 ms` 触发一次 `+P(warm)`，solver objective 从 `1.012` 降到 `0.264`，收益为 `0.748`。
- 真机两域（5090 + 3090a）ShareGPT 600 请求 × 2 轮的六策略对比已完成：完整
  `casr_lp` 的 P95 为 `795.7 ms`，比 SGLang 风格前缀感知路由基线 `cache_aware`
  （`1238.7 ms`）低 `35.8%`，截尾均值低 `16.6%`；经典 `rr` / `load` 基线更低。
  详细设置、路由落点与偏差声明见第 10.5 节与 `docs/异构环境搭建进展.md`。
- **接入 SLO 后结论更强**：3 域、2 轮 × 600 请求、`TTFT ≤ 500 ms & TPOT ≤ 50 ms`，
  `casr_lp` 的 SLO 达标率 `94.1%`、goodput `3.21 req/s`，而 `cache_aware` 只有
  `68.4%` / `2.33 req/s`；且阈值网格的 6 个格点上 `casr_lp` 全部领先。
  算法侧以一阶预测 TTFT + 有限大惩罚 `p_slo` 把 SLO 写进目标函数。
  详见第 10.6 节。

> **⚠️ 2026-09-13 重大更正**：上一条与第 10.5–10.8 节里所有
> `load` / `cache_aware` 的数字都来自一个**缺陷版本**的基线——
> `_pick_load` 用字符串 id 做平局打破，退化成"静态钉死字典序最小的实例"。
> 修复为 `(inflight+1)/capacity` 后重测（wide/deep 各 3 轮）：
> 修复后的基线已经接近最优，**`casr_lp` 在 wide 上略差约 6%、在 deep 上不占优**；
> 只有四域（含慢链路 A100、容量与算力不一致）场景下 `casr_lp` 领先 `26%`。
> 完整口径见 `docs/实验结果汇总.md` §6.0 与 `docs/异构环境搭建进展.md`
> 的"`load` 基线因为字符串平局打破而退化成静态钉死"一节。
- 上面的模拟器结论仍然有效；真机结论的适用范围限于两域、600 请求、`3.5 req/s`
  的 ShareGPT 回放，五节点大拓扑尚未形成统一运行时下的正式结论。

## 2. 测试范围

测试对象包括：

1. Prefix class 哈希、block 对齐及复用状态更新。
2. Cache-aware P/D 容量建模和 min-cost flow/greedy 求解。
3. P-D pair 的 RTT、带宽、Decode queue cost 以及共享链路容量约束。
4. `keep`、`+P(cold)`、`+P(warm)`、`-P` 结构反事实评估。
5. GPU 资源申请、worker warmup、drain、reclaim、release 和 reuse。
6. Prometheus-compatible 遥测解析及外部 `ReconfigExecutor` 命令适配。
7. 低/高/低弹性负载、结构失配和热点漂移实验。

## 3. 环境与复现

- 测试日期：`2026-09-09`
- 运行方式：Python 本地模拟器
- 请求 trace：`workloads/casr_elasticity_low_high_low.jsonl`
- 消融请求数：`111`
- 控制周期：`100 ms`
- 数据类型：`bfloat16`
- KV block size：`16`
- OR-Tools 不可用时使用确定性 greedy；本次结构失配实验显式使用 greedy。

关键命令：

```bash
cd LLMServingSim
python -m unittest tests.test_casr tests.test_workload_generator -v
python tests/run_casr_algorithm_ablation.py \
  --result-dir /tmp/casr-test-report-ablation
bash tests/run_casr_structural_experiment.sh \
  /tmp/casr-test-report-structural
bash tests/run_casr_hotspot_drift.sh \
  /tmp/casr-test-report-drift
python -m py_compile tests/run_casr_algorithm_ablation.py \
  tests/plot_casr_results.py
git diff --check
```

## 4. 自动化测试

| 测试类别 | 结果 | 验证内容 |
|---|---:|---|
| CASR 核心测试 | `15/15` | solver、缓存容量、网络链路、遥测、生命周期、执行器、结构评估 |
| workload generator 测试 | `6/6` | 稳定性、热点选择、泊松到达、低高低阶段、漂移、manifest |
| Python 编译检查 | 通过 | 消融运行器和绘图工具无语法错误 |
| diff 格式检查 | 通过 | 无 whitespace error |

## 5. 算法消融结果

所有算法变体使用同一条低/高/低 workload trace，主要结果如下：

| Case | 平均 latency (ms) | P95 (ms) | 结构动作 | 说明 |
|---|---:|---:|---:|---|
| Static | 929.179 | 1210.060 | 不适用 | 固定结构 baseline |
| CASR | 785.159 | 804.702 | 0 次重复动作 | 完整算法 |
| NoCacheCapacity | 785.159 | 804.702 | 0 | 当前 trace 容量较宽裕 |
| NoWarmCounterfactual | 785.159 | 804.702 | 0 | 当前 trace 未形成 warm 决策差异 |
| NoStructuralGain | 785.159 | 804.702 | 0 | 当前 trace 未形成结构收益差异 |
| NoNetwork | 839.796 | 1126.634 | 0 | 去掉 RTT/带宽/队列网络代价 |
| NoHysteresis | 788.166 | 805.098 | `+P` 4 次 | 去掉收益阈值和 dwell time |

相对 Static，完整 CASR 的平均 latency 降幅为 `15.50%`，P95 降幅为 `33.50%`。NoNetwork 相比完整 CASR 的平均 latency 增加约 `6.96%`，P95 增加约 `40.01%`。NoHysteresis 虽然延迟接近完整 CASR，但额外触发结构动作并产生重复 warm/scale-out 事件，说明仅观察最终延迟不足以评价控制稳定性。

本组消融中 `NoCacheCapacity`、`NoWarmCounterfactual` 和 `NoStructuralGain` 与完整 CASR 数值相同。这不是这些能力已经被证明无效，而是当前 trace 的 P/D 容量和热点分布未制造足够强的瓶颈，下一轮应使用临界容量、共享链路拥塞和热点迁移场景。

## 6. 结构失配实验

配置：`configs/cluster/casr_structural_mismatch.json`。该场景让生命周期的高层容量判断与 flow solver 的当前 P 容量不一致，从而构造“总资源可用但结构不匹配”的反事实场景。

观测结果：

| 指标 | 结果 |
|---|---:|
| 结构动作 | `+P(warm)` 1 次 |
| 决策时间 | 约 `108 ms` |
| base objective | `1.012` |
| candidate objective | `0.264` |
| objective gain | `0.748` |
| warm requested | `8 MiB` |
| 实际新增 KV bytes | `0` |

实际新增字节为 `0` 表示目标 prefix 已驻留，执行的是复用已有缓存的 warm 目标，不应解读为 warmup 失败。动作完成后没有重复结构调整。

## 7. 热点漂移实验

在恒定 `30 req/s` 下，热点从 `hotspot-0` 漂移到 `hotspot-1`：

- 前半段：`hotspot-0` 为 `46/60`，`hotspot-1` 为 `14/60`。
- 后半段：`hotspot-1` 为 `40/60`，`hotspot-0` 为 `20/60`。
- solver objective 从 `0.002` 变化到 `0.0126`。
- 活跃 Prefill 从 `[0]` 变为 `[0, 1]`。
- 未触发额外结构动作。

该结果验证 Prefix class 状态能够随热点变化更新，但由于当前资源容量足够，尚不能单独证明热点漂移一定带来 warm/cold 收益。

## 8. 资源与生命周期观测

完整 CASR 消融运行记录到以下资源事件：

```text
resource_acquire: 5
drain_start: 1
reclaim_scheduled: 1
resource_release: 2
warm_start: 1
warm_complete: 1
deactivate: 1
```

NoHysteresis 运行记录到 `warm_start: 5`、`warm_complete: 4`、`resource_reuse: 4` 和 `executor_scale_out: 4`，与其 4 次 `+P` 动作一致。说明本地资源编排、释放复用和可选外部执行器事件已经进入状态快照。

## 9. 当前限制与下一步

当前测试不能代表完整五节点真实服务性能，主要缺口是：

- 已运行真实 vLLM/LMCache P/D handoff；Prometheus exporter 和真实 serving 指标采集仍未接入。
- warmup 仍是模拟器内 KV 状态操作，未测量真实 GPU/NVMe/网络 I/O 时间。
- Ray 已连接三台 10.212.* 节点并提供资源标签，但尚未把 vLLM 容器生命周期自动编排接入 Ray worker manager。
- 消融 trace 对 cache capacity、warm 和 structural gain 的压力不足。
- 尚未在 8×A100 上完成同模型真实服务；真实模型上的 TPOT、吞吐、SLO attainment、goodput 和 GPU utilization 仍需补齐。

建议下一阶段按以下顺序推进：

1. 在已恢复互信的五节点环境中完成小规模真实 vLLM + LMCache 部署，校准 Prefill/Decode 延迟、KV bytes、RTT 和带宽参数。
2. 构造容量临界、共享 WAN 瓶颈和热点迁移 workload，重复完整消融矩阵。
3. 接入 Prometheus exporter 和 Ray/Kubernetes executor，验证 telemetry 到扩缩容动作的闭环。
4. 在真实集群补充 P50/P95/P99 TTFT、TPOT、吞吐、SLO 和 GPU-seconds 报告。

## 10. 结果文件

本次运行产物位于：

- `/tmp/casr-test-report-ablation/summary.json`
- `/tmp/casr-test-report-structural/casr.jsonl`
- `/tmp/casr-test-report-drift/casr.jsonl`

## 10.1 真实 Qwen3 异构链路验证

2026-09-10 使用真实 RTX5090/RTX4090 profile 运行 `configs/cluster/casr_real_qwen3_tp4_heterogeneous.json`。该配置对应 4×RTX5090 TP4 Prefill、4×RTX4090 TP4 Decode，ASTRA-Sim 拓扑为 `[4,3]`，1 请求 smoke test 完整退出：总输入 64 tokens、生成 127 tokens、TTFT `63.08 ms`、TPOT `19.22 ms`。

随后使用 `tests/run_real_qwen3_hetero_comparison.sh` 完成 16 请求 Static、CASR Greedy 和 CASR LP 快速矩阵，三组均为 16/16 完成，平均 latency `5237.092 ms`、P95 `5279.882 ms`，Decode 实例只有一个，因此三组结果相同。这组实验只证明真实 profile、Qwen3 MoE trace 和 TP4 P/D handoff 可运行，不证明 CASR 的优势。

多候选路由和结构调整的优势证据仍采用旧异构控制矩阵：80 请求实验中 LP 相比 baseline 平均 latency 从 `2067.042 ms` 降至 `1755.377 ms`（`15.1%`），P95 从 `2540.314 ms` 降至 `2256.963 ms`（`11.2%`），Decode 流量由 `40/40` 调整为 `71/9`；Greedy 平均 latency 为 `1844.020 ms`。该矩阵使用已验证的 heterogeneous hardware abstraction，待真实环境增加第二个可用 P/D 候选后复现。

本轮还修复了混合 TP 配置的拓扑问题：原 Qwen3 TP2/TP4 配置总执行 NPU 数为 18，不能用单一矩形 topology 表示 TP2 和 TP4 的不同 collective 组。配置构建器现仅接受可由最大 TP 因子前缀表达的混合 TP，并对不满足条件的配置提前报错。

## 11. 当前复现实验

2026-09-09 在当前工作区重新运行 `tests/run_casr_ablation.sh`，结果产物位于 `/tmp/casr-current-ablation`。本轮使用同一条 `111` 请求低/高/低 workload，固定 2P、固定 1P、动态 CASR（LP/greedy）、仅路由和无 prefix 复用配置均完成，无请求被拒绝。

| 配置 | 高负载平均 latency | 高负载 P95 | 平均 GPU | GPU-seconds | 资源动作 |
|---|---:|---:|---:|---:|---|
| Fixed 2P | `940.161 ms` | `1210.446 ms` | `4.000` | `20.294` | 无 |
| Fixed 1P | `940.061 ms` | `1208.582 ms` | `3.000` | `15.265` | 无 |
| Dynamic CASR (LP) | `790.572 ms` | `805.098 ms` | `3.553` | `16.904` | `1` 次扩容，`2` 次释放 |
| Dynamic CASR (greedy) | `790.572 ms` | `805.098 ms` | `3.553` | `16.904` | `1` 次扩容，`2` 次释放 |
| Routing-only | `790.572 ms` | `805.098 ms` | `4.000` | `18.947` | 无 |

相对 Fixed 2P，当前 Dynamic CASR 的高负载平均 latency 下降约 `15.9%`，P95 下降约 `33.5%`；相对 Routing-only，资源生命周期使 GPU-seconds 下降约 `10.8%`。本轮 LP 与 greedy 结果相同，说明该 workload 的流量规模没有制造求解器差异。无 prefix 复用配置的高负载平均 latency 为 `941.886 ms`、P95 为 `1213.626 ms`，支持 prefix cache 对该 workload 有实际影响的判断。

这轮仍是本地模拟器实验；真实 vLLM 已在 4090、5090 和单卡 3090 分别完成过链路验证，并已完成真实 Qwen3 TP4 跨节点 P/D handoff smoke test。8×A100 节点 `10.70.251.47:2222` 已补齐 Qwen3 TP1/2/4/8 的短上下文 layerwise profile，其中 TP2/4/8 使用 `max_model_len=4096`、attention KV 上限 `1024`，完整 262K 长上下文和 skew 数据仍未纳入精度对比。

## 10.2 真实 Llama 双 P/D 异构对比

使用 `configs/cluster/casr_real_llama_heterogeneous.json` 和 `tests/run_real_llama_hetero_comparison.sh`，在 RTX5090/RTX4090 两个节点各部署一个 TP1 Prefill 和一个 TP1 Decode。80 请求、同一 workload 的结果为：

| 配置 | 平均 latency | P50 | P95 | 平均 TTFT | 平均 TPOT |
|---|---:|---:|---:|---:|---:|
| Static | `1997.493 ms` | `1475.254 ms` | `2541.307 ms` | `29.447 ms` | `15.496 ms` |
| CASR Greedy | `1909.764 ms` | `1482.793 ms` | `2506.271 ms` | `28.249 ms` | `14.815 ms` |
| CASR LP | `1766.501 ms` | `1506.202 ms` | `2432.750 ms` | `28.249 ms` | `13.687 ms` |

相对 Static，Greedy 平均 latency 下降 `4.4%`，LP 下降 `11.6%`；LP P95 下降 `4.3%`，平均 TPOT 下降 `11.7%`。LP 的流量统计为 `P5090→D5090=610.37`、`P4090→D5090=716.51`、`P4090→D4090=375.17`，最终请求级 Decode 分配为 RTX5090/RTX4090=`57/23`；Greedy 触发 3 次扩容、2 次释放，LP 保持 1 个 active Prefill 并完成 1 次资源复用。

本实验是真实 GPU profile 驱动的 ASTRA-Sim 模拟器对比，不是跨主机真实 vLLM/LMCache P/D 服务。RTX5090 Llama profile 当前缺少 `sampler` 行，模拟器跳过该缺失 head 层并告警；补采 sampler 后需要重跑本矩阵。

### 消融结果

同一 80 请求 trace 的 CASR LP 消融如下：

| 变体 | 平均 latency | P95 latency | 说明 |
|---|---:|---:|---|
| 完整 CASR LP | `1766.501 ms` | `2432.750 ms` | prefix cache、网络代价、lifecycle 均启用 |
| no-cache CASR LP | `2062.008 ms` | `2742.863 ms` | 对应 no-cache Static 为 `2092.799 ms` |
| no-network-cost CASR LP | `1659.566 ms` | `2273.661 ms` | 链路 RTT/带宽/共享容量约束移除 |
| no-elasticity CASR LP | `1766.501 ms` | `2432.750 ms` | `max_active_prefill=1` |

no-cache 相对完整 LP 平均 latency 增加 `16.7%`；no-network-cost 比完整 LP 再低 `6.1%`，说明当前实测同网段链路参数是主要成本来源。no-elasticity 与完整 LP 相同，结合 LP 的 lifecycle 日志没有扩容动作，说明该 workload 尚未覆盖结构扩缩容收益；Greedy 则记录 3 次扩容和 2 次释放。消融脚本为 `tests/run_real_llama_hetero_ablation.sh`，结果仍属于 profile 驱动模拟器实验。

## 10.3 Qwen3-8B profile 与真实 P/D 当前状态

为支持单卡/双卡实例，新增 `Qwen/Qwen3-8B` 配置 `configs/model/Qwen/Qwen3-8B.json`。RTX4090、RTX5090 已完成 TP1/TP2 的真实 layerwise profile；RTX3090 已完成 TP1 以及 TP2 的 `dense/per_sequence`，TP2 `attention.csv` 仍在远端采集。统一参数为 `bfloat16`、`max_num_batched_tokens=2048`、`max_num_seqs=128`、`attention-max-kv=16384`，skew 暂未采集。三域配置 `configs/cluster/casr_real_qwen3_8b_three_domain.json` 需待 3090 TP2 attention bundle 完成并通过 profile checker 后再进行正式对比；A100 暂沿用 Qwen3-30B profile，未将不同模型混入对比。

真实跨主机 P/D 入口已切换到 LMCache `enable_pd` + `transfer_channel: nixl` 配置，并新增 `disagg_proxy_pd.py` 注入 `disagg_spec`。2026-09-10 已在 5090→3090 单卡 Qwen3-8B 部署上完成 handoff 验收：代理返回 HTTP `200`，5090 Prefill 日志为 `Stored 34 out of total 34 tokens`，3090 Decode 日志为 `LMCache hit tokens: 34`。UCX 已固定使用 5090 `ens6f0` 与 3090 `ens18`，解决其误选不可达 Calico 地址 `10.42.88.64` 的问题。两端镜像 digest 当前不同（5090：vLLM 0.27.1/LMCache 0.5.3；3090：vLLM 0.29.0/LMCache 0.5.4），因此该结果仅证明真实 handoff，不作为跨版本性能结论；正式性能实验前需统一镜像。

> **⚠️ 2026-09-13 更正：这条"handoff 验收"的说服力不够。** 34 token 不足
> LMCache 的一个 chunk，**根本不会触发真正的 KV 搬运**，Decode 其实是本地
> 算完的。当天发现 `pd_skip_proxy_notification: true` 会让 Decode 在 KV
> 落地前读到上一个请求的槽位（健康检查与 `LMCache hit tokens: N/N` 全部照常
> 通过，延迟也正常，但答案是换行/秒停/乱码）。修复后新增
> `tests/check_pd_correctness.py`：用长文 + 首尾随机校验码判定，
> router 8/8 通过、5 个 P/D 对 10/10 通过，而修复前是 6/6 失败。
> 具体过程见 `docs/五台异构实验环境部署记录.md` 的"P/D 输出正确性缺陷"一节。

### 10.4 当前快速对比结果

在 ASTRA-Sim 重建后，使用 `tests/run_real_llama_hetero_comparison.sh` 对 RTX5090/RTX4090 profile 驱动的双域配置运行 16 请求 Static、CASR Greedy 和 CASR LP，三组均完成 `16/16`：

| 策略 | 平均 latency | 平均 TTFT | 平均 TPOT | Decode 分配 |
|---|---:|---:|---:|---|
| Static | `1841.387 ms` | `32.252 ms` | `14.245 ms` | `8/8` |
| CASR Greedy | `1688.304 ms` | `28.954 ms` | `13.066 ms` | `11/5` |
| CASR LP | `1592.930 ms` | `28.954 ms` | `12.315 ms` | `13/3` |

相对 Static，Greedy 平均 latency 下降 `8.3%`，LP 下降 `13.5%`；LP 平均 TPOT 下降 `13.5%`。同一 workload 的 LP 消融中，no-cache 平均 latency 为 `1620.261 ms`、no-network 为 `1507.313 ms`、no-elasticity 为 `1592.930 ms`。该快速矩阵用于验证实验链路，正式结论仍应扩大请求数并统一镜像/profile 版本。

### 10.5 真机 ShareGPT：CASR vs SOTA 前缀感知路由（2026-09-11）

在真实跨主机 vLLM + LMCache P/D 拓扑上（5090 + 3090a 两域，
`DISABLED_INSTANCES=p3090b,p4090,d4090`），用同一份 ShareGPT 回放 trace
（`workloads/sharegpt-real-qwen3-8b-600-sps14.jsonl`，600 请求，
`3.5 req/s`，并发 `4`，`max_output_tokens=16`）跑完六种策略，每个策略开跑前
重启 P/D 清空缓存并预热全部 P-D 对。表为 **2 轮 × 600 请求** 的合并统计
（`tests/aggregate_real_comparison.py /tmp/casr-md/lp35 /tmp/casr-md/lp36`）：

| 策略 | 对应系统 | 均值 | 截尾均值 | P50 | P95 |
|---|---|---:|---:|---:|---:|
| `rr` | 经典轮询 | `1104.2 ms` | `772.7 ms` | `918.9 ms` | `1564.8 ms` |
| `load` | vLLM least-loaded / shortest-queue | `674.6 ms` | `579.8 ms` | `437.7 ms` | `1276.8 ms` |
| `cache_aware` | SGLang Router / vLLM production-stack 前缀感知路由 | `739.7 ms` | `575.2 ms` | `426.6 ms` | `1238.7 ms` |
| `casr` | 本文启发式层 | `844.9 ms` | `506.0 ms` | `438.1 ms` | `1000.3 ms` |
| **`casr_lp`** | 本文完整慢环 LP | **`487.4 ms`** | **`479.9 ms`** | `443.1 ms` | **`795.7 ms`** |
| `casr_full` | `casr_lp` + `+P/-P` 结构弹性 | `743.5 ms` | `570.6 ms` | `501.3 ms` | `1054.0 ms` |

结论：

1. `casr_lp` 相对 SOTA 路由基线 `cache_aware` **P95 低 `35.8%`**、截尾均值低
   `16.6%`，P50 高 `3.9%`：收益体现在抹平尾部而不是压低中位数。
2. 机制可核验：基线把 `11–23%` 的 Decode 放在比 `d5090` 慢 `2.9×` 的
   `d3090a` 上，而 `casr` 系列的代价模型把 Decode 全部收拢到 `d5090`。
3. 两处必须声明的偏差：(a) 少数请求出现 `47–51 s` 的 LMCache/NIXL 传输毛刺，
   与路由策略无关，因此应固定报告**截尾均值 + P95** 并单列毛刺数；
   (b) 每格仅 2 轮，CI 很宽，优势幅度需补到 3 轮以上。
4. 表中 `casr_full` 的 `743.5 ms` 是**修复前**的数字。根因是控制面对
   `ACTIVE` 实例每 tick 做一次远端 `docker inspect`（~`0.3 s/实例`），
   拉长了采样窗口，而 3% 迟滞又把首版计划冻结整轮。改为按 `state_refresh_s`
   限流探测后，同一 trace 的验证轮里 `casr_full` 为
   `484.0 / 432.7 / 824.2 ms`（均值/P50/P95），与 `casr_lp` 持平。
   修复前后的完整对照见 `docs/异构环境搭建进展.md` 的
   "`casr_full` 目前不加分"一节。

### 10.6 SLO 建模与带约束的真机对比（2026-09-12）

SLO 不再是事后统计口径，而是**进入算法目标函数**：

1. **预测**：`flow_solver._predicted_ttft_ms` 按设计文档 `c_ijk` 的结构预测
   `链路(distance+rtt+transfer) + prefill_service_ms×work_ratio +
   decode_service_ms×queued_fraction`；`work_ratio` 只算未命中前缀，
   decode 项只算排队（TTFT 在首 token 结束）。
2. **惩罚**：预测超过 `ttft_slo_ms` 的三元组单位代价 `+slo_penalty`——
   加有限大惩罚而非删除 pair，容量不足时仍出计划，有达标 pair 时优先选它；
   违约 pair 经 `diagnostics.slo_violating_pairs` 暴露到 `/routing-state`。
3. **测量**：客户端以 SSE 回放，逐请求记 `ttft_ms`/`tpot_ms`；
   SLO 随请求（trace 行或客户端兜底）经 `X-SLO-TTFT-MS`/`X-SLO-TPOT-MS`
   下发，router 把 `slo_ok` 写进 `metrics-<policy>.jsonl`，
   聚合器优先采用请求级判定。

结果（3 域、2 轮 × 600 请求、SLO = `TTFT ≤ 500 ms & TPOT ≤ 50 ms`）：

| 策略 | 均值 | P95 | TTFT P95 | TPOT P95 | SLO 达标率 | goodput |
|---|---:|---:|---:|---:|---:|---:|
| `rr` | `633.4 ms` | `1299.3 ms` | `542.8 ms` | `52.4 ms` | `85.8%` | `2.91 req/s` |
| `load` | `813.4 ms` | `1292.9 ms` | `579.9 ms` | `50.8 ms` | `81.1%` | `2.76 req/s` |
| `cache_aware`（SGLang 风格 SOTA 路由） | `845.3 ms` | `1427.8 ms` | `634.6 ms` | `57.5 ms` | `71.0%` | `2.42 req/s` |
| `casr` | `501.6 ms` | `841.8 ms` | `602.6 ms` | `18.0 ms` | `90.2%` | `3.07 req/s` |
| **`casr_lp`** | `393.7 ms` | `536.1 ms` | `315.6 ms` | `16.7 ms` | **`99.5%`** | `3.39 req/s` |
| `casr_full` | `396.2 ms` | `559.8 ms` | `328.2 ms` | `16.6 ms` | `99.8%` | `3.40 req/s` |

相对 SOTA 路由基线：均值低 `53.4%`、P95 低 `62.5%`、SLO 达标率高 `28.5` 个百分点、
goodput 高 `40%`。阈值网格（`TTFT ∈ {250,500,1000} × TPOT ∈ {25,50}`）六个格点上
`casr_lp` 全部领先，结论不依赖单点阈值。

> 这一版结果包含 2026-09-12 修掉的三处阻塞（TTFT 模型标定、迟滞把陈旧计划锁死、
> Prefill 兜底不看异构），比同日早先那一版（`casr_lp 484.1 / 715.3 ms`，
> SLO `94.1%`）明显更好。**两版不可混用**，细节见
> `docs/异构环境搭建进展.md` 的"三个阻塞点"一节。

声明两点：(a) SLO 的**数值**来自应用假设（同量级于 MLPerf Llama2-70B interactive
档的 `ttft ≤ 450 ms / tpot ≤ 40 ms`），**不是数据集给的**，本仓库的 trace 也未
自带延迟约束；(b) 达标率对 `47–51 s` 传输毛刺极敏感，必须与 `glitch_count`
一起报告。

### 10.7 第二个负载形状：ShareGPT deep 档（强前缀复用）

wide 档的前缀桶复用只有 `1.30`，不能说明"缓存感知路由在真该发挥作用的场景下
是否更好"。因此按 `docs/实验数据集与对比基线说明.md` §2.3 生成 deep 档
（`--max-sessions 200`，会话内多轮，356 请求 / 124 前缀桶 / 桶复用 `2.87`），
同一拓扑、同一 SLO 口径，2 轮共 712 请求：

| 策略 | 均值 | 截尾均值 | P50 | P95 | TTFT P95 | TPOT P95 | SLO 达标率 | >5s 毛刺 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rr` | `705.8 ms` | `696.7 ms` | `432.8 ms` | `1534.6 ms` | `615.2 ms` | `74.1 ms` | `77.7%` | 0 |
| `load` | `891.7 ms` | `883.5 ms` | `1071.9 ms` | `1506.2 ms` | `604.1 ms` | `67.6 ms` | `59.1%` | 0 |
| `cache_aware`（SOTA） | `1291.0 ms` | `1075.8 ms` | `1071.2 ms` | `1696.0 ms` | `784.7 ms` | `83.8 ms` | `52.8%` | 4 |
| `casr` | `508.8 ms` | `502.3 ms` | `437.3 ms` | `947.7 ms` | `729.9 ms` | `17.8 ms` | `86.2%` | 0 |
| **`casr_lp`** | `976.4 ms` | **`549.1 ms`** | `393.2 ms` | **`574.9 ms`** | `351.1 ms` | `17.1 ms` | **`97.8%`** | 8 |
| `casr_full` | `970.3 ms` | `541.4 ms` | `377.4 ms` | `583.9 ms` | `362.1 ms` | `16.7 ms` | `98.3%` | 8 |

要点：(1) **前缀复用变强时 `cache_aware` 反而退化**（SLO 从 wide 档的 `71.0%`
掉到 `52.8%`），而 `casr_lp` 的 P95 反而更好（`715 → 575 ms`）——
说明"把同一前缀都钉到一个 Prefill"并不天然更优，还要看该实例的
落盘/推送能力；(2) `casr_lp`/`casr_full` 各有 `8/712` 个 `~51 s` 传输毛刺，
原始均值被抬高，必须与截尾均值、毛刺数一起读。

### 10.8 第二个数据集：Databricks Dolly-15k（零前缀复用）

ShareGPT 两个档都带前缀复用，"前缀亲和"与"异构感知"的贡献分不开。
Dolly-15k 是单轮指令数据，**600 请求 / 600 个唯一前缀（桶复用 1.00）**、
输入均值 `193` token（ShareGPT wide 是 `971`），来源与定制见
`docs/实验数据集与对比基线说明.md` §5。拓扑为 Prefill `p5090/p4090` +
Decode `d5090/d3090a/d4090`（3090a 主机驱动/库不匹配，其 Decode 用
`PRESERVE_INSTANCES` 保留）。2 轮 × 600 = 1200 请求：

| 策略 | 均值 | 截尾均值 | P50 | P95 | TTFT P95 | TPOT P95 | SLO 达标率 | goodput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rr` | `543.3 ms` | `537.1 ms` | `370.8 ms` | `1046.9 ms` | `237.8 ms` | `53.4 ms` | `91.2%` | `12.71 req/s` |
| `load` | `838.1 ms` | `831.6 ms` | `910.5 ms` | `1306.6 ms` | `284.5 ms` | `67.7 ms` | `68.3%` | `6.48 req/s` |
| `cache_aware`（SOTA） | `845.9 ms` | `839.0 ms` | `917.6 ms` | `1223.9 ms` | `282.4 ms` | `67.0 ms` | `66.2%` | `6.23 req/s` |
| `casr` | `363.8 ms` | `361.6 ms` | `365.6 ms` | `433.4 ms` | `179.2 ms` | `19.0 ms` | `99.8%` | `14.28 req/s` |
| **`casr_lp`** | `354.9 ms` | `353.0 ms` | `345.8 ms` | `449.6 ms` | `214.9 ms` | `17.8 ms` | **`99.5%`** | `14.23 req/s` |
| `casr_full` | `357.9 ms` | `356.0 ms` | `346.5 ms` | `452.8 ms` | `215.2 ms` | `17.6 ms` | `99.3%` | `14.22 req/s` |

对 SOTA 路由基线：均值低 `58.0%`、P95 低 `63.3%`、SLO 达标率高 `33.3` 个百分点、
goodput 高 `128%`。

**机制**：`load` 与 `cache_aware` 把 `76–80%` 的 Decode 流量放到
`d3090a`（实测 `service_ms 421.6`，是 `d5090` 的 `2.9×`）——因为它最闲；
`casr_lp`/`casr_full` 则是 `100%` 落在 `d5090`。由于该负载**零前缀复用**，
这一优势只能来自异构感知，不能归因于缓存命中。
