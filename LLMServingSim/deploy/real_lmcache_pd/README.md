# 跨主机真实 vLLM + LMCache P/D 部署

一套"每台机器一个域、每域 1 Prefill + 1 Decode"的真实分离式推理栈：
vLLM 容器负责 Prefill / Decode，LMCache 负责跨主机 KV 搬运，一个 FastAPI
router 把请求按策略分派到不同的 (Prefill, Decode) 对。

上层还有一层 Ray，只做**资源发现与域标签**，不参与推理调度
（见 `../ray/README.md`）。

## 组成

| 文件 | 作用 |
|---|---|
| `router_config.json` | 拓扑与代价参数的唯一来源：`hosts`（ssh/模型/网卡）、`prefills`、`decodes`、`links`、`weights`、`casr` |
| `start_multidomain_pd.sh` | 按上面的配置渲染每个实例的 LMCache YAML，ssh 推到目标主机，起 host-network 的 vLLM 容器 |
| `disagg_router.py` | 请求级 router：选 Prefill、选 Decode、可选地做结构动作（`+P/-P`），并写出 per-request metrics |
| `disagg_proxy_pd.py` | 单机 P/D 代理（早期形态，保留作参照） |
| `casr_control.py` | 与模拟器**共用**的 CASR 控制循环（LP 求解、亲和计划、生命周期） |
| `router_image.Dockerfile` | router 镜像：vLLM 镜像 + OR-Tools（LP）+ openssh-client（`+P/-P`） |
| `lmcache-{prefiller,decoder}-crosshost.yaml` | 单机 P/D 的 LMCache 配置模板 |

## 起停

```bash
# 1) 起/重建整张 P/D 拓扑（会先删掉同名容器再重建）
bash deploy/real_lmcache_pd/start_multidomain_pd.sh

# 只重启某个域 / 先看计划不执行
ONLY=a100 bash deploy/real_lmcache_pd/start_multidomain_pd.sh
DRY_RUN=1 bash deploy/real_lmcache_pd/start_multidomain_pd.sh   # 计划写到 /tmp/casr-multidomain-plan.sh

# 2) 打 router 镜像（vLLM + OR-Tools + openssh-client）
docker build -f deploy/real_lmcache_pd/router_image.Dockerfile \
  -t casr-router:latest deploy/real_lmcache_pd

# 3) 起 router（容器，host network；策略在启动时固定）
docker run -d --name casr-md-router --network host \
  -v "$PWD/deploy/real_lmcache_pd:/router-config:ro" \
  -v "$PWD/serving:/casr-serving:ro" \
  -v /tmp/casr-md-results:/results \
  -e PYTHONPATH=/casr-serving \
  --entrypoint python3 \
  casr-router:latest /router-config/disagg_router.py \
    --config /router-config/router_config.json \
    --policy casr_lp --scale-backend "" --port 9000 \
    --metrics /results/metrics-casr_lp.jsonl

# 4) 健康检查
curl -s localhost:9000/health                 # {"status":"ok","policy":"casr_lp"}
curl -s localhost:9000/routing-state | python3 -m json.tool | head -30
```

`--policy` 可选 `rr | load | random | cache_aware | casr | casr_full | casr_lp`
以及三个消融 `casr_noservice / casr_noaffinity / casr_nonetwork`。
`--scale-backend docker` 才开启 `+P/-P`（需要 router 容器能 ssh 到各 worker）。

## 落盘约定：日志/缓存都不进系统盘（2026-09-15）

`hosts.<domain>` 里新增三个路径，`start_multidomain_pd.sh` 按它们渲染启动脚本：

| 键 | 用途 |
|---|---|
| `log_dir` | `<instance>.sh`（启动脚本）、`<instance>.run.sh`（起停包装）、`<instance>.log`（vLLM 输出）、`<instance>.pid` |
| `cache_dir` | vLLM 编译缓存（`VLLM_CACHE_ROOT`，docker 侧挂到 `/root/.cache/vllm`） |
| `ray_temp_dir` | Ray 会话目录与日志（见 `../ray/README.md`） |

每台机指向本机数据盘：4090 用 `/mnt/home/casr`、5090 用 `/data/casr`、
3090a 用 `/data/sdb/model/casr`、3090b 用 `/data/casr`、a100 用
`/mnt/adminserver-nfsrdma/casr`。docker 域用 `--log-driver=none` 把 stdout 直接
重定向到 `log_dir` 里的文件，`/var/lib/docker` 不再累积容器日志（3090a 根盘只有 8.8 GB）。

`launcher` 决定怎么起 vLLM，per-host 或 per-instance 都可覆盖：

| `launcher` | 行为 |
|---|---|
| `docker`（默认） | host-network `docker run`，镜像由 `image` 决定 |
| `venv` | 直接跑 `<venv_python>` 同目录下的 `vllm`，需要同时给 `venv_python`。A100 域用它（该机 `buaa` 没有 docker 组也不在 sudoers，`guest` 只允许公钥登录） |

`DRY_RUN=1` 只生成 `/tmp/casr-multidomain-plan.sh` 不执行，改启动参数前先看一遍；
生成的 `<log_dir>/<instance>.sh` 就是该实例的完整启动命令，可直接复现。

## `start_multidomain_pd.sh` 的环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `CONFIG` | `router_config.json` | 拓扑来源 |
| `IMAGE` | `vllm/vllm-openai:casr029` | `casr029` = vLLM `0.29.0` + LMCache `0.5.4` |
| `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` / `MAX_NUM_SEQS` | `0.80` / `4096` / `16` | 必须与 capacity 标定时的取值一致，否则 `capacity` 失效 |
| `PD_BUFFER_SIZE` | `1073741824`（1 GiB） | LMCache 的 PD staging 缓冲，**必须覆盖同时在途的 KV**：按 `147 KB/token` 估算，`2048 token × 8 并发 ≈ 2.4 GiB` |
| `PD_BUFFER_DEVICE` | `cuda` | 长 trace 建议 `cpu`：放 GPU 上会在几百个唯一前缀后耗尽，接收端永久卡在 `Failed to allocate memory object` |
| `PD_SKIP_PROXY_NOTIFICATION` | `0` | **不要设成 1**。设 1 时 Prefill 不等接收端通知就返回，而 LMCache 接收端在**分配槽位时**就把 chunk key 写进索引，Decode 会读到上一个请求留下的数据 → 返回极快的垃圾（换行/立刻 EOS）。7 个实例都开着它时，router 的 5 个长文请求有 3 个答不出自己 prompt 里的校验码；关掉后 8/8 通过。当初设它是为了配合 LMCache `async` backend 的 ZMQ notification，我们现在用 `sync`，不需要它 |
| `PD_PROXY_HOST` / `PD_PROXY_PORT` | 空（不启用） | Prefill sender 的 ZMQ 通知端点（router 的 `KVLandingTracker` 绑定 PULL）。**两个都必须设**：sender 的 `pd_proxy_host is not None` 断言在缺省时会失败，LMCache 引擎直接标 `init failed` → 完全不缓存不搬运（输出仍然正确，因为 Decode 自己重算，但等于 2× prefill 计算）。启用后 KV 真的会搬，但 2026-09-13 实测收到的 KV 用不了（四对全乱码），详见部署记录 |
| `KV_TRANSFER_BACKEND` | `lmcache` | `native` = 改用 vLLM 自带的 `NixlConnector`（`KV_TRANSFER_BACKEND=native`）。这是 2026-09-13 唯一**既正确又真的搬 KV** 的路径：连接器自报 `176 MB/次、106–343 MB/s`。native 模式下每个实例会自动分配唯一的 `VLLM_NIXL_SIDE_CHANNEL_PORT`（默认 5600 会同主机冲突）并把 `VLLM_NIXL_SIDE_CHANNEL_HOST` 播成本机地址 |
| `KV_NATIVE_ALLOW_MIXED` | `0` | 设 1 会加 `enforce_handshake_compat: false`。**实测无效**：0.26↔0.29 连元数据都解不开（`missing field block_strides`）。跨版本只能升级镜像，不能靠关检查 |
| `LOCAL_PREFILL` | `auto` | **CASR 的"搬不搬 KV"决策**：`auto` 按代价模型逐请求比较"搬运"（`intercept + slope×tokens`，按 §5.14 实测标定）与"Decode 本地重算"（`93 ms/1k token ÷ speed` + 各自的队列项），取小者；`never` 恢复"总是搬运"（做 A/B）；`always` 强制本地重算 |
| `LOCAL_PREFILL_BASELINES` | `never` | 基线策略（`rr`/`load`/`random`/`cache_aware`）默认**不做**这个决策，保持经典 P/D"总是搬运"的行为——这正是 CASR 要对比的系统。设 `auto` 可让基线也获得同样能力（公平性消融用） |
| `LOCAL_PREFILL_MS_PER_1K` / `TRANSFER_MS_PER_1K_LOCAL` / `TRANSFER_MS_PER_1K_CROSS` / `TRANSFER_FIXED_MS_CROSS` | `93` / `585` / `1351` / `48` | 代价模型的四个实测参数（`tests/calibrate_native_pd.py` 标定，2026-09-13）。换硬件/拓扑后必须重标，否则决策会偏 |
| `DISABLED_INSTANCES` | 空 | 本次运行跳过的实例 id（逗号分隔），用于临时故障节点；router 与健康门禁同样遵守 |
| `PRESERVE_INSTANCES` | 空 | **不重建**、只要求健康的实例 id。用于"主机驱动漂移、旧容器还能跑但新容器起不来"的节点 |
| `ONLY` | 空 | 只重建部分实例 |

### router 侧的运行参数（由 run 脚本透传给容器）

| 变量 | 默认（配置里的值） | 说明 |
|---|---|---|
| `SLO_TTFT_MS` / `SLO_TPOT_MS` / `SLO_PENALTY` | `500` / `50` / `10` | 请求级 SLO 与惩罚项；`SLO_PENALTY=0` 但仍统计，可做"测量 vs 算法使用"的消融 |
| `OVERFLOW_PENALTY` | `10` | 计划里"单位容量超卖"的单价。饱和档默认值偏小，可用来把计划压成容量可行 |
| `PLAN_TTL_S` | `4` | 计划有效期。默认 4 s 下饱和档实测有 `70/600` 请求撞上"计划已过期"而落到另一套落点规则；调大可减少这类抖动 |
| `PREFILL_CLASS_LIMIT` | 空 | 单个 Prefill 的 class 工作集上限 |
| `EQUALIZE_{PREFILL,DECODE}_CAPACITY` | 空 | 消融开关：把该角色所有实例的容量拉平，使剩下的差异只剩 `service_ms` |

> 拓扑里的 **A100 节点统一用 `10.66.0.15:2222`**（`10.70.251.47` 是同一台机器
> 的旧路由地址，已不再被任何工具引用）；任何手工 ssh 都要带 `-p 2222`，
> 漏了会看到 `Connection refused` 而不是认证失败。另外这台机器**必须用
> `~/.ssh/id_casr_cluster`**（非默认命名的密钥，已在 `~/.ssh/config` 里为
> `10.66.0.15` 配好 `IdentityFile`）；用默认密钥会看到
> `Permission denied (publickey,password)`，这不是公钥没加，而是密钥没被选中。
> HTTP 健康检查仍走 `8100/8200`，且这两个地址都不在 `no_proxy` 里，
> 人工 `curl` 记得加 `--noproxy '*'`（脚本用 `trust_env=False`，测量不受影响）。
>
> `links` 里的 a100 条目已于 2026-09-13 在新路径上重测（iperf3 单流，
> 6 s，跳过前 1 s）：RTT 全对 `41.5 ms`，带宽 `a100→5090 115 Mbps`、
> `5090→a100 96 Mbps`、`3090a 147/86`、`3090b 142/94`、`4090 140/101 Mbps`。

## 一次完整的对比实验

`tests/run_real_multidomain_comparison.sh` 是唯一推荐入口，它按顺序做：

1. **Ray 域门禁**：`deploy/ray/domain_inventory.py --check --skip-health`
   核对 `router_config.json` 里的域与 Ray 上注册的 `domain:*` 标签是否一致
   （只比名字会掩盖改名，脚本按键值比对）。
2. **重建 + 健康门禁**：`start_multidomain_pd.sh` 起全部启用实例，轮询
   `/health` 直到全绿（最多 3 次重建，容忍 GPU 显存释放竞态）。
3. **全对预热**：对**每一个** (P, D) 组合发两个 `do_remote_decode` 请求。
   首次握手要付一次性 ~50 s 的 JIT 编译代价，不预热会让第一批请求全是超时值。
3.5 **输出正确性门禁**：`tests/check_pd_correctness.py` 对一个长文发
   `Access code A/B` 口令题，答不出自己的码就判失败，失败超过
   `CORRECTNESS_MAX_FAIL_RATIO`（默认 `0.25`）直接中止本轮，产物写
   `correctness-<policy>.txt`。延迟门禁抓不到"快速返回的垃圾"，这一步才是
   证明 handoff 真的生效的判据。
4. **逐策略测量**：每个策略都回到第 2 步重建（保证冷缓存），再跑 trace。
5. 产物写进结果目录：`metrics-<policy>.jsonl`（per-request）、
   `summary-<policy>.json`、`state-<policy>.json`、`client-<policy>.jsonl`、
   `topology.json`、`run-config.json`。

```bash
cd LLMServingSim
POLICIES="load cache_aware casr casr_lp casr_full" STREAM=1 \
  SLO_TTFT_MS=500 SLO_TPOT_MS=50 SLO_PENALTY=10 \
  TRACE=workloads/dolly-real-qwen3-8b-600-sps14.jsonl \
  NUM_REQS=200 MAX_OUTPUT_TOKENS=16 CONCURRENCY=8 TIME_SCALE=1.0 \
  DISABLED_INSTANCES=p3090b,p3090a PRESERVE_INSTANCES=d3090a \
  PD_BUFFER_SIZE=34359738368 PD_BUFFER_DEVICE=cpu \
  bash tests/run_real_multidomain_comparison.sh /tmp/casr-md/<tag>
```

多轮结果用聚合器出表（含跨轮 95% CI、截尾均值、SLO 达标率、goodput、
流量落点与毛刺计数）：

```bash
python3 tests/aggregate_real_comparison.py /tmp/casr-md/<tag>-r1 \
  /tmp/casr-md/<tag>-r2 /tmp/casr-md/<tag>-r3 --baseline load --pairs \
  --slo-ttft-ms 500 --slo-tpot-ms 50
```

## 已知坑（每条都真实踩过）

- **健康检查通过 ≠ P/D 真的在工作**：`pd_skip_proxy_notification: true` 时
  Decode 会在 KV 落地前读到上一个请求的槽位，日志照样打
  `LMCache hit tokens: N/N`、延迟只有 `~300 ms`，但输出是换行/秒停。
  每个 policy 跑测之前必须过 `tests/check_pd_correctness.py`（run 脚本已内建，
  见 `correctness-<policy>.txt`）——它用长文末尾的随机校验码做判定，
  答不出就是错的。
- **`PD_PROXY_PORT` 必须在 router 和实例两边一致**：router 绑了 PULL 而实例
  没配 `pd_proxy_host`（或反过来）时，每个请求都会卡到 `PD_PROXY_WAIT_S`
  （默认 60 s）超时。run 脚本从同一个环境变量派生两边，手工起容器时容易踩。
- **"输出正确"不等于"KV 搬过去了"**：现在的默认配置（不给 proxy）就是
  Decode 自己重算 prompt，输出全对但完全没有 offload。判断是否真的 offload
  要看 Decode 日志的 `LMCache hit tokens`（为 0 就是没搬）和 `/routing-state`
  的 `kv_landings` 计数，不能只看延迟或正确性门禁。
- **native 模式下 P/D 必须同版本**：vLLM 0.26 与 0.29 的 NIXL 元数据不兼容
  （`compatibility hash mismatch` → 关掉检查后变成
  `missing required field block_strides`）。`router_config.json` 的
  `hosts.<domain>.kv_group` 标记版本组，router 只保留实例最多的一组并在启动
  时打印丢弃了哪些实例（当前 `v029` 组，`p_a100/d_a100` 被排除）。
  A100 要入网得先把镜像升到同一版 vLLM。
- **native 模式的真实传输指标**在实例日志里：`KV Transfer metrics: Num
  successful transfers=… Avg MB per transfer=… Throughput (MB/s)=…`，
  这是标定链路代价最直接的输入（跨主机 10.212 实测 `106–343 MB/s`）。
- **`capacity` / `speed` 必须实测**：这些字段早期是"按硬件比例声明"的，
  A100 的 prefill capacity 声明 `550`、实测只有 `51.3` req/s（`10.7×`），
  最小负载基线因此把 95% 的 prefill 压到最慢的机器上，对比会失真。
  用 `tests/calibrate_instance_capacity.py --role prefill|decode
  --endpoints name=host:port,...` 实测，再按比例回填（保留一个实例做标尺）。
- **结果目录不要复用**：`client-<policy>.jsonl` / `metrics-<policy>.jsonl` 是
  **追加**写的，聚合器按 `request_id` 去重时保留**最先出现**的行，复用目录会把
  上一轮的旧数据混进来（2026-09-13 踩过一次：`dolly-r1` 里混着 7 小时前的
  600 行）。run 脚本现在发现非空结果目录就直接报错退出；确需复用时显式设
  `ALLOW_RESULT_REUSE=1`。
- **别绕过 router 直接给 Prefill 发普通请求**：`kv_producer` 角色的实例收到
  不带 `kv_transfer_params` 的请求会停在 `Running: 1 reqs`，之后该实例不再推进，
  只能重建容器。所有校验都走 `:9000` 或 `tests/calibrate_real_pd.py`。
- **`pd_buffer_size` 不够会假死**：接收端反复打印
  `pd_backend.py:939 Failed to allocate memory object, retrying...`，
  整轮从第 ~300 个请求开始全是超时。长 trace 一律
  `PD_BUFFER_SIZE=34359738368 PD_BUFFER_DEVICE=cpu`。
- **首个 P-D 握手偶尔死锁**：两个 EngineCore 100% 空转、请求不返回。
  预热脚本对每个 pair 设 `WARMUP_TIMEOUT`（默认 300 s）并整体重来，
  不要让整个 batch 挂死。
- **驱动漂移节点**：`unattended-upgrade` 换掉内核模块后，主机 `nvidia-smi`
  报 `Driver/library version mismatch`；已运行的容器仍可服务，新容器起不来。
  这类实例用 `PRESERVE_INSTANCES=<id>` 保住，恢复需要重启那台机器。
- **代理**：控制面 `http(s)_proxy` 指向 `10.212.70.196:12345`，会拦内网地址；
  压测脚本用 `httpx.AsyncClient(trust_env=False)` 绕过，`curl` 依赖 `no_proxy`。

节点清单、事故记录与当天可用拓扑见
`docs/五台异构实验环境部署记录.md`；实验结果见 `docs/实验结果汇总.md`。
