# CASR 算法量化设计（2026-09-23）

> **这份文档回答一个问题**：CASR 到底在优化什么、每个参数取多少、每个数从哪来。
> 一切以代码为准（`serving/casr/flow_solver.py`、`serving/casr/evaluator.py`、
> `serving/core/hw_service.py`、`serving/core/router.py`），常数都标了来源：
> **实测**（哪次校准）、**bundle 推导**（哪份 profile）、或**配置**（哪个 json）。
>
> 分工：[CASR核心算法设计.md](CASR核心算法设计.md) 讲为什么这样做，本文讲**算出来是多少**；
> 实验数字见 [实验结果汇总.md](实验结果汇总.md)。

---

## 0. 一句话

在「前缀亲和类 k × Prefill i × Decode j」上求连续流量 `f[i,j,k]`，
最小化「**网络搬运 + 排队 + 算力 + 拥塞 + 超容**」的加权和；
再在同一套目标上做**一步结构反事实**（±P），只有收益超过阈值才动结构。

控制面默认每 1 s 求一次（`control_interval_s`），计划 TTL 4 s，
需求用 EWMA（`ewma_alpha = 0.2`）估计，命中率半衰期 10 s。

---

## 1. 符号

| 符号 | 含义 | 单位 | 来源 |
|---|---|---|---|
| k | 请求类：前缀指纹 × 输出长度桶（类名形如 `...\|out:16-31`） | — | router 分桶 |
| i, j | Prefill / Decode 实例 | — | 集群配置 |
| **f[i,j,k]** | 类 k 走 (i,j) 的流量 | req/s | **决策变量** |
| λ_k | 类 k 的到达率（EWMA） | req/s | 观测 |
| P_i, D_j | Prefill / Decode 单请求服务时间 | ms | bundle 推导（§5） |
| K^P_i, K^D_j | Prefill / Decode 容量（参考长度请求/秒） | req/s | bundle 推导（§5） |
| w[i,k] | 类 k 在 Prefill i 上要算的**参考长度份数** | 无量纲 | `miss × 长度因子` |
| o_k | 类 k 在 Decode 上的参考长度份数 | 无量纲 | 输出长度 / 参考长度 |
| b_k | 类 k 一条请求的 KV 字节数 | byte | `kv_bytes_per_token × prompt_tokens` |
| C_l | 共享链路 l 的字节预算 | B/s | 实测链路 |
| S_j | Decode 的 `max_num_seqs` | 条 | 配置 |
| Q_j, R_j | Decode 的 waiting / running 数 | 条 | 观测 |
| p_ovf | 超容罚系数（`overflow_penalty`） | 代价/容量比例 | 配置 = 10 |

**w[i,k] 的定义**（`_cache_work`，两因子相乘）：

```text
w[i,k] = max(0.05, 1 - hit[i,k])          # 未命中的部分
       × clamp(t_k · K^P_i / T_i, 1, 8)   # 长度因子（上限 work_ceiling = 8）
```

其中 `t_k` 是类 k 的 prompt 长度，`T_i` 是该卡实测的 **prompt 处理上限**（tokens/s；
真实部署 p5090 14000、p4090 7300、p3090 5600）。没有 token 上限标定时退化为
「固定 + 线性」模型 `(4.4 + 11.2·t_k/1000) / (4.4+11.2)`——拟合自 p5090 上
180 token 156 req/s 与 1024 token 62.9 req/s 两点。

> 这一项是 2026-09-14 补的：在那之前 1250-token 请求和 200-token 请求对 Prefill 是**同一个价**，
> 于是 Prefill 排队 48 深、`prefill_ms` P95 = 34 s 时 LP 仍报 `prefill_overflow = 0`。

---

## 2. 单条边的代价 c[i,j,k]

`_pair_cost()`，四项相加，单位统一为**秒**：

```text
c[i,j,k] = w_net · ( δ_ij + rtt_ij + b_k / BW_ij )          # 网络
         + w_q   · ( 4·Q_j + R_j ) / S_j                    # 排队
         + w_c   · ( P_i · w[i,k] + D_j · o_k ) / 1000      # 算力（ms -> s）
         + 1[ TTFT_hat(i,j,k) > SLO_k ] · p_slo             # SLO 软罚
```

| 项 | 代码 | 量纲 | 真实部署取值 | 仿真 P-15B 取值 |
|---|---|---|---|---|
| δ_ij | `(start_npu_i − start_npu_j) × 0.001` | s | 0（同域）/ 0.001·Δ（跨域） | 同 |
| rtt_ij | `pair_costs[i,j]["rtt_ms"] / 1000` | s | 实测 48 ms（跨域） | 由链路模型给 |
| b_k / BW_ij | `class_kv_bytes / bandwidth_bytes_per_s` | s | 184 MB / 260 MB/s ≈ 0.71 s（Qwen3-8B） | 21.2 MB / 链路带宽 |
| w_q | `queue_weight` | — | **1.0** | 0（仿真用容量表征排队） |
| w_c | `compute_weight` | — | **1.0** | **1.0** |
| w_net | `network_weight` | — | **1.0** | **1.0** |
| p_slo | `slo_penalty` | 代价单位 | **10.0** | 0（未启用） |
| SLO_k | `ttft_slo_ms` | ms | **500** | 0（未启用） |

**为什么算力项必须存在**：容量约束只写得下「能不能」，写不下「多贵」。异构集群里 5090 比
3090 快 3 倍，没有这一项时慢卡会变成「便宜的溢出下水道」——实测 `casr_lp` 把 95% 的类堆在
最便宜的一台上，反而输给 `load` 基线 6%。

### 2.1 TTFT 预测（`_predicted_ttft_ms`）

```text
q_j        = ( 4·Q_j + R_j ) / S_j
TTFT_hat   = 1000·δ_ij + rtt_ms + transfer_ms
           + ( P_i · w[i,k] + ovh_prefill[i] )
           + ( ttft_decode[j] + D_j · o_k · q_j )
```

| 固定开销 | 含义 | 实测（真实部署 2026-09-12） |
|---|---|---|
| `prefill_overhead_ms[i]` | 写 KV 那部分（TTFT − prefill_ms） | p5090 73 ms、p4090 52 ms、p3090b 184 ms |
| `decode_ttft_ms[j]` | 空队列时 Decode 也要付的一次开销 | d5090 56 ms、d4090 163 ms |

> 不加这两项时「空集群」预测 TTFT ≈ 50 ms，实测 ≈ 240 ms，**任何现实 SLO 都不会触发**。
> 它们只用于判断是否越过 SLO，不参与其它决策。

---

## 3. 线性规划（`solver: "lp"`，OR-Tools GLOP）

### 3.1 变量与约束

```text
min  Σ_{k,i,j} f[i,j,k] · c[i,j,k]
   + p_ovf · ( Σ_i σP_i + Σ_j σD_j + Σ_l σL_l )
   + 拥塞项（见 3.2）

s.t. Σ_{i,j} f[i,j,k] = λ_k                              ∀k     需求守恒
     Σ_{k,j} f[i,j,k]·w[i,k] / K^P_i ≤ 1 + σP_i          ∀i     Prefill 容量
     Σ_{k,i} f[i,j,k]·o_k     / K^D_j ≤ 1 + σD_j          ∀j     Decode 容量
     Σ_{k,(i,j)∈l} f[i,j,k]·b_k / C_l ≤ 1 + σL_l          ∀l     链路字节预算
     f ≥ 0,  σ ≥ 0
```

**溢出按容量比例计价**，不按原始单位：一字节/秒的链路溢出比一次算力项大 10^8 倍，
按原始单位计会让 LP 为微小溢出做大迁移、丢掉已预热的 KV。

### 3.2 拥塞项：分段线性凸价（`_congestion_variables`）

把容量 K 均分 m 段（`utilization_segments = 8`），第 t 段的边际价

```text
p_t = w_u · s · (2t + 1) / (2m),     s = 该实例的服务时间（秒）
```

于是负荷 L 的总价正好是 **w_u · s · L² / (2K)**：第一单几乎免费、最后一单值一整次服务时间。
这正是纯线性价格缺的那半个信号——没有它，LP 会把所有类堆在最便宜的那一台上。

### 3.3 后处理三条（LP 之后再走）

| 规则 | 触发条件 | 动作 | 参数（真实部署值） |
|---|---|---|---|
| `_collapse_low_demand` | λ_k < `single_home_below_rps` | 把该类流量合并到 LP 最偏好的 Prefill（除非会造成溢出） | **1.0 req/s** |
| `_cap_class_footprint` | 某 Prefill 的类数 > `prefill_class_limit` | 逐个搬走「最好搬」的类直到回到上限 | 默认 0（关） |
| `_enforce_egress_budget` | 某生产者的链路字节预算被超过 | 整类搬到还有余量的最便宜对 | `shared_links[*].capacity_bytes_per_s` |

> 第三条为什么必要：LP 的链路 slack 是 soft 且**按类聚合**的，单个类流量太小、溢出不显形。
> 实测（2026-09-16 六域 arena）：LP 自报 `link_overflow = 0`，而 A100 推送链路实际被超 **11×**、
> 284/376 请求落在那里，p95 37 s。把 `overflow_penalty` 提高 100× 也没修好（284 → 280）。

### 3.4 计划迟滞

已发布的计划用**同一目标**重新打分（`plan_objective`）：

* 计划没覆盖到的类按 `plan_uncovered_penalty`（默认 **10**）每单位需求计费——
  否则「只写第一个 tick 见过的类」的计划对后来的类看起来**免费**，迟滞会把它永远留着；
* 新计划只有在相对改善超过 `plan_gain_threshold_rel`（**0.03**）时才替换。

---

## 4. 结构决策 ±P（`serving/casr/evaluator.py`）

### 4.1 判据

```text
gain(+P) = H_eff · ( J_base − J_cand − h ) − startup_cost,   H_eff = max(0, H − startup_s)
gain(−P) = H      · ( J_base − J_cand + h )                  # 删除立即生效，全窗口计收益
```

接受条件（两个都要过）：

```text
gain > gain_threshold_abs
且   gain / max( H·|J_base| , demand_scale ) > gain_threshold_rel
```

* `H` = `structural.evaluation_window_ms`：编辑必须在多长的过载里回本
* `h` = 每台活跃 Prefill 的持有成本（`holding_cost`，或 `idle_cost_fraction` × 饱和算力成本）
* `J_base` = 当前结构的计划目标值；`J_cand` = 反事实结构的计划目标值

### 4.2 参数与它们的物理含义

| 参数 | 真实部署 | 仿真 P-15B 配置 | 含义 / 为什么是这个数 |
|---|---:|---:|---|
| `evaluation_window_ms` | **60000** | 1000 | 用 1 s 时「重启容器」永远回不了本，判据退化成「有空闲就开」（实测：几乎空闲时 +P 也在 tick 0 触发，gain 0.0375，而真实峰值下同一反事实值 ~10/s） |
| `startup_s` | **45** | 0（默认） | 实测容器重启到可用 45 s；+P 收益只算 `H − 45`，它同时是**最短冷却**，防止决定在自己启动期内被反转 |
| `idle_cost_fraction` | **0.1** | 0（默认） | 每台活跃 Prefill 每秒付「饱和算力成本的 10%」。没有持有成本时 −P 的反事实机械为负，池子只会涨不会缩（实测 −P 收益 −1.25 / −0.56 / −10.09 / −10.20） |
| `gain_threshold_abs` | **0.001** | 0.001 | 绝对门槛，滤数值噪声 |
| `gain_threshold_rel` | **0.01** | 0.01 | 相对门槛（相对 H·J_base） |
| `dwell_time_ms` | **3000** | 1000 | 两次结构编辑最小间隔，且强制 ≥ `startup_s` |
| `max_active_prefill` | **5** | 3 | 上限；统计「还在付 GPU 的 worker」（含 WARMING/DRAINING），只数 ACTIVE 会超配 |
| `prescale_utilization` | 0.8 | 0.8 | **预测式扩容**：负载到池子实际推送能力的 80% 就先起备用机。反事实式扩容在 90 s 峰值里付不起 45 s 启动（实测六域：反应式 +P 停在 14572 ms，按 egress 提前开的池子 1120 ms） |
| `enable_warm_counterfactual` | **false** | true | 是否考虑「先预热再切」（额外 `warm_cost`，按 `warm_top_k` / `warm_budget_bytes` 估预热收益） |

### 4.3 −P 的额外闸门

删一台只在**池子整体空闲**且候选实例没有在飞请求（`waiting` / `running` / `inflight` 全空）时允许。
实测（2026-09-16 小集群）：1.05 req/s 轻载下每台在两个请求之间都会瞬间空闲，逐候选检查不够，
控制器把 p5090 删了两次、池子两次塌掉。

---

## 5. 参数从哪儿来：profile → 算法

### 5.1 单步成本（`hw_service.step_cost_ns`）

```text
step(hw, model, tp, t) = Σ_prologue dense(t)
                       + Σ_每层 ( Σ_该层的 dense 项(t) + moe(t) )
                       + Σ_head per_sequence
```

逐项从 `profiler/perf/<HW>/<model>/<variant>/tp<N>/*.csv` 按 t 查表；
**attention 被刻意排除**（随上下文变化，1k 上下文下约 0.01 ms，比权重读小三个数量级），
所以「两张卡的速度比」由权重读决定——这正是集群 TPOT / prefill 时间量到的量。

### 5.2 服务时间与容量（`rescale_service_times` / `rescale_capacities`）

```text
P_i = P_anchor · step(i) / min_k step(k)        # 最快的卡保持配置锚点，其余按步时比值缩放
K^P_i = 1000 / step(i, t=1024)                  # 参考长度请求/秒
K^D_j = 1000 / ( step(j, t=1) × 16 )            # 16 输出 token 为参考
```

### 5.3 P-15B 三域（3090 + 4090 + 5090）现算值（2026-09-23）

| 实例 | 卡 | prefill_service_ms / prefill_capacity (req/s) | decode_service_ms / decode_capacity (req/s) |
|---|---|---:|---:|
| 0 / 1 | RTX3090 | 884.8 / **29.48** | 772.5 / **4.50** |
| 2 / 3 | RTX4090 | 424.8 / **61.14** | 372.9 / **9.32** |
| 4 / 5 | RTX5090 | 290.9 / **89.42** | 278.3 / **12.49** |

同一配置在 **tp=2**（bundle `bf16/tp2`）：prefill 863.4 / 420.5 / 290.9 ms，
decode 763.4 / 352.8 / 278.3 ms——只有 4090 与 5090 的 decode 明显变化，
与「MLA latent 读不随头切分」的实测一致。

**KV 记账**：`kv_bytes_per_token = 16960`（P-15B 设计闭式 16.56 KB/token），
对比 Qwen3-8B 的 147456，差 **8.9×**——这一项直接决定链路上的 `b_k/BW` 与预算是否绑得住（§6.2）。

---

## 6. 量化行为（LP 诊断工具现算）

工具：`tests/diagnose_lp_terms.py`（离线重建求解器、逐项打印）。
场景：三域 P-15B 拓扑、8 类、1250-token prompt、容量取 §5.3 实测值、链路 2 GB/s/生产者。

### 6.1 随负载上升：先绑住的是 Decode

| 提供负载 | 目标值 | prefill 溢出 | decode 溢出 | 链路溢出 |
|---:|---:|---:|---:|---:|
| 8 req/s | 5.36 | 0 | 0 | 0 |
| 16 req/s | 11.89 | 0 | 0 | 0 |
| 32 req/s | 34.15 | 0 | **0.55** | 0 |
| 64 req/s | 91.05 | 0 | **3.11** | 0 |
| 128 req/s | 212.36 | 0.004 | **8.24** | 0 |
| 192 req/s | 347.08 | **0.55** | **13.36** | **0.47** |

合计 decode 容量 = 4.50 + 9.32 + 12.49 = **26.3 req/s**；prefill 合计 180 req/s；
链路合计 6 GB/s（≈ 283 req/s）。所以在 P-15B 上**瓶颈是 Decode**——
这也解释了为什么「加一个 Prefill」在很多档里救不了场：+P 只加 Prefill。

### 6.2 同负载、只换 KV：链路约束的开关

| 提供负载 | P-15B（16,960 B/token）链路溢出 | Qwen3-8B（147,456 B/token）链路溢出 |
|---:|---:|---:|
| 64 req/s | 0 | **2.90**（p2 链路被超 2.9×） |
| 128 req/s | 0 | **1.56 / 7.24** |
| 192 req/s | **0.47** | **2.52 / 12.17** |

单条链路预算 2 GB/s，一条请求的 KV：P-15B 21.2 MB → **94 req/s**；
Qwen3-8B 184 MB → **10.9 req/s**。**KV 压缩 8.9× 把「每台生产者能推多少」抬高 8.9×**，
于是 P-15B 上链路要到 ~190 req/s 才成为约束，而 Qwen3-8B 在 64 req/s 就已经压爆。

---

## 7. 实测收益（同一套参数跑出来的）

| 场景 | 基线 | CASR | 备注 |
|---|---:|---:|---|
| 三域 1250-token 8 rps，**P-15B** | load 486 ms | **354 ms（−27%）** | TTFT p50 696 → 76 ms |
| 三域 1250-token 8 rps，**Qwen3-8B** | load 205,062 ms | **84,839 ms（−59%）** | KV 压死链路，调度可捡的份额更大 |
| 五域 32 rps（过载），P-15B | casr_lp 64,106 ms | **casr_full 58,199 ms（−9.2%）** | 静态池过载时 +P 才有正收益 |
| 五域 8 rps（未饱和），P-15B | casr_lp 313 ms | casr_full 326 ms（**+4%**） | 唤醒的第三台只分到 84 个请求，却付了 45 s 冷启动 |
| 三域 tp1 → tp2，P-15B | 403.96 ms | **340.27 ms（−15.8%）** | 调度份额 −7.5% → −4.8% |

**读法**：硬件 / 切分 / KV 压缩越强，绝对性能越好，但「调度还能加多少」的**相对**份额越小——
这五行是同一结论的五个切面。

---

## 8. 参数清单（默认 / 真实部署 / 仿真）

| 参数 | 默认 | 真实部署 | 仿真 P-15B | 作用 |
|---|---:|---:|---:|---|
| `solver` | greedy | lp | lp | 求解器 |
| `control_interval_s` / `plan_ttl_s` | 1 / 2× | **1 / 4** | 1（sim）/ — | 控制频率与计划寿命 |
| `ewma_alpha` / `hit_half_life_s` | — | **0.2 / 10** | — | 需求与命中率估计 |
| `overflow_penalty` | 10 | 10 | 10 | 超容罚 |
| `utilization_weight` / `utilization_segments` | 0 / 8 | **1 / 8** | **1 / 8** | 凸拥塞价 |
| `network_weight` | 1.0 | 1.0 | **1.0** | 网络项权重 |
| `queue_weight` | 0 | **1.0** | 0 | 排队项权重 |
| `compute_weight` | 0 | **1.0** | **1.0** | 算力项权重 |
| `ttft_slo_ms` / `slo_penalty` | 0 / 0 | **500 / 10** | 0 / 0 | SLO 软罚（0 = 不建模 SLO） |
| `capacity_reference_tokens` | 1024 | 1024 | 1024 | Prefill 容量参考长度 |
| `decode_reference_tokens` | 16 | 16 | 16 | Decode 容量参考长度 |
| `single_home_below_rps` | 0（关） | **1.0** | 0（关） | 低需求单宿 |
| `prefill_class_limit` | 0（关） | 0 | 0 | 每台 Prefill 的工作集上限 |
| `class_demand_floor_rps` | 0 | 0 | 0 | 在飞类的需求地板 |
| `work_ceiling` | 8.0 | — | 8.0 | 长度因子上限（防 token 解析错） |
| `plan_uncovered_penalty` | 10 | 10 | 10 | 未覆盖类罚 |
| `plan_gain_threshold_rel` / `_abs` | — | **0.03 / 0** | — | 计划替换门槛 |

---

## 9. 复现

```bash
cd LLMServingSim

# 1) 集群配置 -> probe 配置（内部会 chdir 到 astra-sim 解 bundle，并打印
#    从 profile 反推的 service_ms / capacity，即 §5.3 的表）
python3 tests/make_lp_probe_config.py \
    --cluster-config configs/cluster/casr_p15b_three_domain.json \
    --prompt-tokens 1250 --output /tmp/p15b_probe.json

# 2) 逐项分解 LP 目标：SLO 关、KV 用本模型的 16960 B/token
python3 tests/diagnose_lp_terms.py --config /tmp/p15b_probe.json \
    --rate 32 --classes 8 --prompt-tokens 1250 --kv-bytes-per-token 16960 \
    --prefills p0,p2,p4 --decodes d1,d3,d5 --flows
# 只换 KV（Qwen3-8B 的 147456）就能复现 §6.2 的链路溢出

# 3) 端到端三臂（baseline / greedy / LP）
CLUSTER_CONFIG=configs/cluster/casr_p15b_three_domain.json \
    bash tests/run_casr_comparison.sh /tmp/casr-3dom
```

**口径提醒**：`prefill_capacity` / `decode_capacity` / `*_service_ms` 在实验里
**每次都会被 bundle 覆盖**（`rescale_*`），配置里写的只是锚点。
引用「容量」的结论必须说明是「配置锚点」还是「bundle 反推值」。
