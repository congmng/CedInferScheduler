# 真·混合注意力模型接入模拟器（Zamba2-1.2B，2026-09-16）

前面用的混合档是**合成**的：`configs/model/Qwen/Qwen3-8B-hybrid.json` 只改了
`kv_geometry`，所有层跑的仍是 Qwen3 的同一套 kernel。这一份是**真模型**：
Zyphra **Zamba2-1.2B**（Mamba2 + 共享注意力），vLLM 0.29 的 `zamba2.py` 支持。

结论先说：**混合模型的层不是同一种形状**，"少几层 KV"这种一句话的描述根本
不够——32 层只跑 Mamba、6 层是 Mamba+共享注意力，成本结构和 KV 结构都不一样。
模拟器为此加了逐层 block type，算法在这上面的表现与全注意力模型**不是一组
可比的数**。

## 1. 为什么是它

* `config.json` 的 `layers_block_type` 是 `['mamba'×5, 'hybrid']×6 + ['mamba'×2]`：
  **38 层里只有 6 层带 KV**（第 5/11/17/23/29/35 层），其余 32 层是 Mamba2
  混合器（递归状态，与 prompt 长度无关）。
* 与仓库既有意图一致：`deploy/real_lmcache_pd/router_config_zamba.json` 里就写着
  `model_name = zamba2-7b`（7B 权重在 HF 上是 gated，1.2B 同族且开放，架构形状相同）。
* **不需要下载权重**：profiler 以 `load_format=dummy` 起引擎，只需要
  `configs/model/Zyphra/Zamba2-1.2B.json`（HF config）+ 架构 yaml。
  权重只用于核对数字：`model.safetensors` 头里是 1 215 064 704 个参数。

## 2. 产物

| 文件 | 作用 |
|---|---|
| `configs/model/Zyphra/Zamba2-1.2B.json` | HF config + `torch_dtype` + `kv_geometry`（6 个 KV 层 + Mamba 态折合 65 token） |
| `profiler/models/zamba2.yaml` | 架构目录 + 新的 `layer_types`（`mamba` / `hybrid` 两条流水线） |
| `profiler/perf/RTX4090/Zyphra/Zamba2-1.2B/bf16/tp1/` | profile bundle（dense / per_sequence / attention / meta） |

## 3. 怎么跑

```bash
cd LLMServingSim
docker run --rm --gpus '"device=0"' --entrypoint python3 \
  -v "$PWD":/work -w /work vllm/vllm-openai:casr029 \
  -m profiler profile Zyphra/Zamba2-1.2B --hardware RTX4090 --tp 1 \
  --dtype bfloat16 --out-root /work/profiler/perf \
  --measurement-iterations 3 --attention-max-kv 16384
```

**关于 attention 网格（重要，省时间）**：默认 attention 网格在
`max_num_seqs=256 / max_num_batched_tokens=2304 / max_kv=4096` 下是 **4166 个
shot**，而 profiler 每个 shot 有 **~3.5 s 的固定开销**（引擎侧
`layerwise_profile` 每个 shot 进一次 `torch.profiler` 上下文，与形状无关；
实测：MNBT=32/MSQ=2 的 20 个 dense shot 也是 3.5 s/个）。4166 shot ≈ **4 小时**。

因此 attention 单跑用了更粗的网格：

```bash
python3 -m profiler slice Zyphra/Zamba2-1.2B --tp-refresh 1 --group attention \
  --hardware RTX4090 --tp 1 --dtype bfloat16 --out-root /work/profiler/perf \
  --measurement-iterations 1 \
  --attention-max-kv 4096 --attention-chunk-factor 4.0 --attention-kv-factor 4.0
```

860 个 shot ≈ 50 分钟，覆盖 `chunk ∈ {16,64,256,1024,2304}`、`kv ∈ {16,64,256,1024,4096}`
（kv 轴取 4096 与取 2048 的 shot 数一样——`num_cache_tokens` 过滤掉了
大 kv 配大 batch 的组合——所以直接顶到 `max_model_len`）。meta.yaml 会记下
`chunk_factor: 4.0 / kv_factor: 4.0`，谁看都知道这一档是省时间换的。

## 4. 测到的层成本（RTX4090、bf16、tp1）

| 层 | tokens=1（decode 步） | tokens≈1024（prefill） |
|---|---:|---:|
| `mamba_mixer`（Mamba2 混合器，整块） | 87.5 µs | 453.3 µs |
| `qkv_proj` | 107.6 µs | 650.6 µs |
| `o_proj` | 19.7 µs | 109.8 µs |
| `gate_up_proj` | 38.0 µs | 238.3 µs |
| `down_proj` | 38.4 µs | 223.7 µs |
| `shared_linear`（把注意力输出投回 Mamba 路径） | 10.7 µs | 58.8 µs |
| `attention_norm` / `mamba_norm` / `final_layernorm` | 1.7–1.9 µs | 4.2–6.1 µs |

两件对调度有意义的事：

1. **Mamba 层不是"免费"的**：一个 Mamba2 混合器在 decode 步要 87 µs，和同一层的
   `qkv_proj`（108 µs）一个量级；prefill 成本随 token 线性增长（453 µs @1024）。
   混合模型省的是 **KV 显存与搬运**，不是算力。
2. **一次前向只有 6 次注意力**：按 `step_cost_ns` 折算，Zamba2-1.2B 在 4090 上
   decode 步 **4.86 ms**、1024-token prefill **25.5 ms**；同一张卡上的
   Qwen3-8B 是 **25.6 ms / 151.9 ms**（约 5–6 倍）。

## 5. 模拟器为此补了什么

原来 `trace_generator` 只会按**一个** `sequence` 模板重复 `num_hidden_layers` 次。
对 Zamba2 那等于让 38 层都跑完整的共享注意力块，注意力被算 38 次而不是 6 次。
补齐的部分：

* `profiler/models/zamba2.yaml` 新增 `layer_types`，`mamba` 层走
  `[mamba_norm, mamba_mixer]`，`hybrid` 层走
  `[attention_norm, qkv_proj, rotary_emb, attention, o_proj, shared_linear,
  mamba_norm, mamba_mixer, gate_up_proj, act_fn, down_proj]`。
  profiler 的 `Architecture` pydantic 模型同步加了 `layer_types` 字段并在
  load 时校验（写错的层名在 profile 之前就会报错）。
* `trace_generator._block_type_for` / `_type_sequence` / `_is_windowed_layer`：
  按模型 config 的 `layers_block_type[i]` 选流水线；`kv_geometry` 支持
  `full_layer_indices`（KV 层是**交错**的，不是前 N 层）。
* `memory_model.calculate_sizes` 认识 Zamba2 专有的层名
  （`mamba_norm` / `attention_norm` / `mamba_mixer` / `shared_linear`），并且
  接受 `attention_head_dim`（HF 的另一种写法）与 `attention_hidden_size`
  （注意力通路是 4096 宽而 hidden 只有 2048——这点弄错 qkv 的权重会差一倍）。
* `hw_service.step_cost_ns`（控制面定价用的那一步）也走同一份逐层流水线，
  否则"计划"按 38 个注意力块估容量、"时序"按 6 个跑，同一台机器两个数。

## 6. KV 与容量：混合不等于"省一半以上"

| 模型 | 每 token KV（层 × 每层） | 1250-token prompt 的 KV |
|---|---|---:|
| Qwen3-8B（全注意力） | 36 × 4 KB | 184 MB |
| Qwen3-8B-hybrid（合成） | 9×4 KB + 27×256 窗口 | 74 MB |
| **Zamba2-1.2B（真）** | 6×16 KB + 32 层常数态（65 token 折合） | **157 MB** |

关键：Zamba2 带 KV 的 6 层是 **MHA**（`kv_dim = 32×128 = 4096`，每 token 16 KB），
是 Qwen3-8B GQA 层（4 KB）的 4 倍。层数少了 6 倍，每层宽了 4 倍，**净省只有
0.85×**——而不是"38 层变 6 层"暗示的 0.17×。合成档之所以看起来省那么多，
是因为它把窗口层当成几乎不占 KV。

这也是"kvcache 感知调度"要拿真模型验证的原因：**KV 的省法取决于
（带 KV 的层占比）×（每层 KV 宽度）**，两个都随架构变。

## 7. 已知近似

* `zamba2.yaml` 里 `MambaMixer2` 作为一整行绑定（引擎里是融合算子，拆开计时会重复），
  `Zamba2AttentionDecoderLayer` 的两个 RMSNorm 都落在 `attention_norm` 上被取平均。
* 混合层内 MLP 的发射顺序比真实前向晚两格（顺序只影响 Chakra 图里的排布，
  每个算子仍然只收一次费）。
* 权重：checkpoint 只实例化**一份**共享 transformer（第 5 层），其余 5 个 hybrid 层
  通过 rank-128 adapter 复用它；模拟器按"6 个独立 hybrid 层"求和，得到约 1.66B 参数
  （真实 1.215B）。偏大只会让显存上限检查更严格。
* attention 表用的是 4 倍粗网格（见第 3 节）。
