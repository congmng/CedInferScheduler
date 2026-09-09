# CASR 存算协同调度测试报告

## 1. 测试结论

本次测试针对当前 `LLMServingSim` 中的 CASR 调度链路进行验证。结果表明：

- 自动化回归测试 `21/21` 通过，覆盖 Prefix class、缓存容量、流量守恒、网络代价、遥测、资源生命周期、结构反事实和 workload 生成器。
- 在固定低/高/低负载 trace 上，完整 CASR 的平均 latency 为 `785.159 ms`，峰值阶段 P95 为 `804.702 ms`。
- 相比固定结构 baseline，完整 CASR 平均 latency 降低约 `15.5%`，P95 降低约 `33.5%`。
- 去掉网络代价后，平均 latency 上升到 `839.796 ms`，P95 上升到 `1126.634 ms`，说明 RTT、带宽和队列代价会影响路由结果。
- 去掉迟滞机制后触发 `4` 次 `+P`，完整 CASR 未重复执行结构动作，验证了收益阈值和 dwell time 的抑制作用。
- 结构失配场景在约 `108 ms` 触发一次 `+P(warm)`，solver objective 从 `1.012` 降到 `0.264`，收益为 `0.748`。
- 当前结果仍属于模拟器验证。尚未在真实 vLLM、LMCache、Prometheus、Kubernetes 或 Ray 集群上完成端到端测试。

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

当前测试不能代表真实服务性能，主要缺口是：

- 未安装并运行真实 vLLM、LMCache 和 Prometheus exporter。
- warmup 仍是模拟器内 KV 状态操作，未测量真实 GPU/NVMe/网络 I/O 时间。
- 资源编排尚未连接真实 Kubernetes/Ray worker manager。
- 消融 trace 对 cache capacity、warm 和 structural gain 的压力不足。
- 尚未报告真实模型上的 TPOT、吞吐、SLO attainment、goodput 和 GPU utilization。

建议下一阶段按以下顺序推进：

1. 在小规模真实 vLLM + LMCache 环境校准 Prefill/Decode 延迟、KV bytes、RTT 和带宽参数。
2. 构造容量临界、共享 WAN 瓶颈和热点迁移 workload，重复完整消融矩阵。
3. 接入 Prometheus exporter 和 Ray/Kubernetes executor，验证 telemetry 到扩缩容动作的闭环。
4. 在真实集群补充 P50/P95/P99 TTFT、TPOT、吞吐、SLO 和 GPU-seconds 报告。

## 10. 结果文件

本次运行产物位于：

- `/tmp/casr-test-report-ablation/summary.json`
- `/tmp/casr-test-report-structural/casr.jsonl`
- `/tmp/casr-test-report-drift/casr.jsonl`

