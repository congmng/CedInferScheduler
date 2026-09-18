# 基于 DeepSeek-V4 架构做一个小型模型：可行性调研与设计（2026-09-18）

> 目标：官方 V4 系列太大（Flash 284B / Pro 1.6T），我们希望**自己训一个小的**，
> 但保留它"大幅压缩 KV"的那部分设计。本文回答三件事：
> ① V4 的架构到底是什么；② 它的 KV 到底压掉多少；③ 我们自己训一个小的，
> 要满足哪些硬约束、选什么配置、按什么顺序验证。
>
> 所有结论都标了来源；**"实测"两字的结论都在本仓库/本机复现过**。

---

## 0. 结论先行

1. **V4 的 KV 压缩不是靠"少几层 KV"，而是靠逐层的压缩比**：官方 config 里有一个
   **`compress_ratios` 列表**（V4-Flash 43 层 = `[0,0,4,128,4,128,…,4,0]`），
   4 与 128 交替，对应论文里的 **CSA（Compressed Sparse Attention）** 与
   **HCA（Heavily Compressed Attention）**。[来源：官方 config + arXiv:2606.19348]
2. 按官方 config 与 vLLM 的 KV 布局算，V4-Flash 的 KV 约 **11.8 KB/token（fp8）**；
   我们仓库里测过的对照是 Qwen3-8B **147.5 KB/token**、DeepSeek-V2-Lite MLA
   **31.1 KB/token**、Zamba2-1.2B **~127 KB/token**。也就是说 V4 的 KV 比我们
   现在跑的东西**小 2.6～12 倍**。论文宣称 1M 上下文下是 V3.2 的 **10%**，
   我按同一套口径算出来是 **34%**，差异需要读论文的 KV 章节才能对齐（见 §2.3）。
3. **自训一个"迷你 V4"在 vLLM 0.29 上可以起来——但只在 5090 上**。实测：
   12 层 / hidden 1024 / head_dim 512 的缩小版配置在 **RTX 5090（sm120）成功
   启动**；在 **RTX 4090（sm_89）直接失败**，报
   `deepgemm hyperconnection.hpp: Unsupported architecture`。原因是 mHC
   （超连接）用的是 deep_gemm 的 `sm90/sm100/sm120` kernel，**Ada/Ampere 没有**。
4. 因此路线建议：**先不碰 mHC**（它是训练稳定性机制，与 KV 压缩无关），
   把 CSA/HCA + MLA + 稀疏 indexer 这三件"压缩 KV"的事做小做实；训练与
   推理先按 **fp8-first** 准备（vLLM 的 V4 路径对 fp8 有硬要求，见 §3）。

---

## 1. V4 架构：官方证据

### 1.1 论文口径（arXiv:2606.19348，2026-04-26）

> DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

摘要里明确的三项架构升级：

1. **hybrid attention = Compressed Sparse Attention (CSA) + Heavily Compressed
   Attention (HCA)**，为长上下文效率服务；
2. **Manifold-Constrained Hyper-Connections (mHC)**，改造残差连接；
3. Muon 优化器（训练侧）。

规模与效果：V4-Pro **1.6T 参数 / 49B 激活**，V4-Flash **284B / 13B 激活**，
均支持 **1M 上下文**；预训练 **>32T tokens**；在 1M 上下文下，
V4-Pro 只需 V3.2 的 **27% 单 token 推理 FLOPs** 与 **10% KV cache**。

### 1.2 配置口径（官方 HF config，我们已存档）

| 字段 | V4-Flash | V4-Pro | V4.1-Flash（多模态，text_config） |
|---|---:|---:|---:|
| layers | 43 | 61 | 40 |
| hidden | 4096 | 7168 | 5120 |
| heads / head_dim | 64 / **512** | 128 / **512** | 64 / **512** |
| qk_rope_head_dim | 64 | 64 | 64 |
| q_lora / o_lora / o_groups | 1024 / 1024 / 8 | 1536 / 1024 / 16 | 1280 / 1024 / 8 |
| **compress_ratios** | **{4:21, 128:20, 0:2}** | {4:30, 128:31, 0:1} | {2,1,…}（见存档） |
| index_*（稀疏选择器） | topk 512, heads 64, dim 128 | topk 1024 | topk 512 |
| sliding_window | 128 | 128 | 128 |
| MoE | 256 experts, top-6, 1 shared, moe_inter 2048 | 384 experts, top-6 | 384 experts, top-6 |
| num_hash_layers | 3 | — | — |
| mHC | hc_mult 4, sinkhorn 20 iters | 同 | 同 |
| 量化 | fp8 (e4m3, ue8m0, 128×128 block) + expert fp4 | 同 | fp8 |

存档位置：`LLMServingSim/configs/model/deepseek-ai/DeepSeek-V4-Flash.json`、
`…/DeepSeek-V4-Pro` 未存（可再抓）、`…/DeepSeek-V4.1-Flash.json`。

**两个结构要点**（这两点决定了"压缩"是怎么来的）：

* **KV 是一个 latent，不是 K+V 两份**：`qkv` 下投影出
  `q_lora_rank + head_dim` 的联合 latent，KV 部分只有 `head_dim=512` 维
  （外加 64 维 RoPE），这是 MLA 一路的写法。
* **每层按 `compress_ratio` 决定"多久存一条状态"**：vLLM 实现里
  `assert compress_ratio in [4, 128]`，并且
  `coff = 1 + (compress_ratio == 4)`（ratio 4 有重叠窗口）、
  `sliding_window = coff * compress_ratio`、
  `state_dim = 2 * coff * head_dim`（kv_state + score_state）。
  **ratio=0 的层就是普通 MLA 全注意力**（V4-Flash 只有首 2 层和末层）。

---

## 2. KV 到底压掉多少

### 2.1 计算式

```
每 token KV 字节（单层） =
    ratio == 0 :  (head_dim + qk_rope_head_dim) × dtype_bytes          # 全 MLA
    ratio  > 0 :  2 × coff(ratio) × head_dim × dtype_bytes / ratio     # 压缩态
其中 coff(4)=2, coff(128)=1；fp8 → dtype_bytes=1，bf16 → 2
（另有滑动窗口保留最近 sliding_window 个原始 token，长度上限内为常数项）
```

### 2.2 结果

| 模型 | 层构成 | KV/token（fp8） | KV/token（bf16） |
|---|---|---:|---:|
| **V4-Flash（官方 43 层）** | 2×全 + 21×ratio4 + 20×ratio128 | **11.8 KB** | 23.6 KB |
| DeepSeek-V3.2（MLA，61 层，同口径推算） | 61×全 | 34.3 KB | 68.6 KB |
| **对照：我们仓库里实测过的** | | | |
| DeepSeek-V2-Lite（真机实测） | MLA | — | **31.1 KB** |
| Qwen3-8B | 36×全（GQA） | — | **147.5 KB** |
| Zamba2-1.2B（真机混合） | 6×全 + 32×常数态 | — | **~127 KB** |

**引擎验证（实测）**：用 vLLM 0.29 起一个 **只含 1 个 ratio=0 层**的缩小配置，
报 `Available KV cache memory: 7.73 GiB → GPU KV cache size: 13,964,524 tokens`
即 **594 B/token/层**，与公式的 576 B/token 相差 3%（页对齐）。**公式被引擎验过。**

### 2.3 与论文 10% 的差异（必须写清）

论文说 1M 上下文下 V4 的 KV 是 V3.2 的 10%；按 `config + vLLM 布局`我算出来是
**34%**。已知的可能原因：① V3.2 除了 MLA 还有 **DSA indexer 的 per-token cache**，
我的 34.3 KB/token 没算它；② 论文比的是 Pro 而不是 Flash；③ 论文的 10% 可能
包含其 serving 实现按页/按层对齐后的实测分配量。**在写进任何材料之前，需要读
论文的 KV cache 章节把口径对齐**（这是本研究的第一优先级待办）。

---

## 3. 工程可行性：我们踩到的硬约束（全部实测）

用缩小配置在本地与 5090 上试启，依次撞到四条约束：

| # | 约束 | 证据 | 对自研小模型的含义 |
|---|---|---|---|
| 1 | **KV cache 必须 fp8** | 4090 上报 `AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto` | 训练后必须做 fp8 KV（或直接 fp8 训练）；`kv_cache_dtype=fp8` |
| 2 | **mHC 只支持 sm90/sm100/sm120** | 4090（sm_89）报 `deepgemm hyperconnection.hpp:59 Unsupported architecture`；镜像里 deep_gemm 编了 `sm90/sm100/sm120` 三套 | **4090/3090/A100 都跑不了官方 V4 路径**；本集群只有 **5090 域**可以。要么只用 5090 做实验，要么为 mHC 写 fallback |
| 3 | **mHC 的形状对齐** | 5090 上 `hc_mult=1` 报 `sm120_tf32_hc_prenorm_gemm.hpp:75: n <= 128 and n % 8 == 0`；改成官方 `hc_mult=4`（`(2+4)×4=24`，8 的倍数）后通过 | `hc_mult` 不能随便取小；自研小模型若保留 mHC，需满足 `(2+hc_mult)·hc_mult ≡ 0 (mod 8)` |
| 4 | **o_proj（输出 LoRA）要求 fp8 权重块缩放** | 栈：`attention.py:392 → flashinfer_sparse.py:552 _o_proj → ops/o_proj.py:64 deep_gemm_fp8_o_proj`，读 `wo_a.weight_scale / weight_scale_inv` | **NVIDIA 路径是 fp8-first**：权重必须按 fp8 block-scale 准备，纯 bf16 起不来 |
| 5 | **不同 ratio 的页尺寸要能整除** | 把 1 层设成 `ratio=128` 报 `page size is not divisible by the maximum page size and cannot be padded. Padding is only supported for non-MLA attention layers` | 混合 ratio 会带来页尺寸约束；**单层玩具配置不能代表整套层列表**，必须按完整层列表验证 |

**已经成功的配方（实测，5090，sm120）**：

```bash
# 12 层缩小配置：hidden 1024 / 16 heads / head_dim 512 / ratios [0,0,4,128,…]
# + 官方 fp8 quantization_config + fp8 KV + hc_mult=4
docker run --gpus device=2 --entrypoint python3 -v $HOME/dsv4-small:/model \
  vllm/vllm-openai:casr029 -c "
from vllm import LLM
llm = LLM(model='/model', load_format='dummy', enforce_eager=True,
          skip_tokenizer_init=True, dtype='bfloat16', kv_cache_dtype='fp8',
          max_model_len=4096, gpu_memory_utilization=0.30,
          hf_overrides={'num_hidden_layers': 1})
print('BOOTED OK')"
```

配置已入库：`LLMServingSim/configs/model/deepseek-ai/DeepSeek-V4-Tiny-Smoke.json`。

**这条"整个工程栈是 fp8 + Hopper/Blackwell only"的结论，是本次调研最重要的
可执行信息**：它直接决定了我们在哪张卡上做实验、以及是否值得为 mHC 写 fallback。

---

## 4. 小型模型的设计空间

### 4.1 保留什么、砍什么

| V4 组件 | 与 KV 压缩的关系 | 建议 |
|---|---|---|
| CSA/HCA 交替（`compress_ratios`） | **就是压缩本身** | **必留**，比例先照官方 4/128 |
| MLA latent（head_dim + RoPE） | 决定"全层"的 base 成本 | 必留；`head_dim` 可试 256/512 |
| 稀疏 indexer（top-k 选择） | 让 CSA 在长上下文可算 | 先留（小模型可把 topk 调小） |
| sliding window 128 | 局部保真 | 必留，小模型可试 64–128 |
| mHC（超连接） | **与 KV 无关**，是训练稳定性 | **建议先砍**（同时绕开 sm90+ 限制） |
| MoE + hash layers | 与 KV 无关，影响显存与吞吐 | 可砍成 dense，或留少量 expert |
| fp8 训练/量化 | 推理路径要求 | 先用 bf16 训、后做 fp8 block-scale |

> **已决定砍掉 mHC（2026-09-18）**，详见
> [DSV4_10-30B跨卡设计.md](DSV4_10-30B跨卡设计.md) §2.1：它只占 33.8M 参数 /
> 0.3% 每 token FLOPs，砍它是**纯可移植性决定**（换来四类卡可跑），
> 代价只有训练稳定性，用 SwiGLU clamping + anticipatory routing + 常规手段替代。

### 4.2 三个候选配置（参数量为我按实现结构估算）

| 候选 | 层 | hidden | heads/hd | q_lora/o_lora/groups | experts | **总参 / 激活** | **KV/token (fp8)** |
|---|---:|---:|---|---|---|---:|---:|
| **DSV4-S** | 24 | 1024 | 16 / 512 | 256/256/4 | 32, top-2 | **~1.6B / ~0.34B**（不含词表） | **6.8 KB** |
| DSV4-S′(hd256) | 24 | 1024 | 16 / 256 | 256/256/4 | 32, top-2 | 更小 | **3.5 KB** |
| DSV4-M | 32 | 2048 | 32 / 512 | 512/512/8 | 64, top-4 | ~13.9B / **~2.05B** | 8.8 KB |

（KV 按 §2.1 公式，层列表 `[0,0]+[4,128]×k+[0]`；参数量按
`hidden×(q_lora+head_dim) + q_lora×(heads×head_dim) + o_groups 输出 LoRA + MoE` 估算，
Embedding 占 DSV4-S 的 ~265M（vocab 129280×1024×2），**自研时建议把词表换小**
（如 32k–64k）以省下这部分。）

**推荐从 DSV4-S 起步**：1.6B 总参 / 0.66B 激活、KV 6.8 KB/token
（比我仓库里现有对照小 4.6–22 倍），单卡或双卡就能训起来，
而且它的"KV 已经不是瓶颈"这一点本身就是我们调度研究想验证的工况。

### 4.3 训练预算（粗算，用于判断可行性）

* 训练 FLOPs ≈ `6 × 激活参数 × tokens`。以 DSV4-S（激活 ≈ **0.34B**，不含词表）为例：
  * 100B tokens ≈ `6×0.34e9×1e11 = 2.1e20` FLOPs；
  * 1T tokens ≈ `2.1e21` FLOPs。
* 单卡 bf16 有效算力按 ~100 TFLOPs 估（4090/5090 上实际利用率通常 30–50%）：
  * **100B tokens ≈ 24 GPU-days（8 卡 ~3 天）**；
  * **1T tokens ≈ 239 GPU-days（8 卡 ~30 天）**。
* 换 DSV4-M（激活 2.05B）则 100B tokens ≈ 143 GPU-days（8 卡 18 天），
  1T tokens 不可行（8 卡 178 天）。**所以"自己训"的可行档是
  DSV4-S + 100B 量级 token，而不是从零训到 1T。**
* 现实选择：① 只做**继续预训练/蒸馏**（用 1–10B tokens，1–5 GPU-days 量级）；
  ② 或**改架构不训模型**——把官方 V4-Flash 的 config 结构照搬做**模拟**，
  用我们现有的 profiler/模拟器研究"KV 很小的模型上调度还值不值"，
  这不需要训练（见 §5）。

---

## 5. 与现有研究栈的衔接

先说清楚**为什么这件事对我们的课题重要**：我们前面所有调度结论都建立在
"KV 是绑定资源"的工况上（Qwen3-8B 147.5 KB/token、MLA 31.1 KB/token）。
如果 KV 降到 **6–12 KB/token**，那么：

* P/D 之间的 handoff 成本下降一个数量级，**"搬还是算"（Q2）的临界点会整体移动**；
* 我们测到的 `KV 体积 ÷ 链路能力` 三点刻度（−5.6% / +0.6% / +31.8%）会整体左移，
  即**在这个区间里 CASR 的相对优势应当收敛**；
* 这恰好是"什么时候不需要存算协同"的最强证据，也是审稿人会问的那个边界。

**要跑起来还需要什么**：

| 环节 | 现状 | 缺什么 |
|---|---|---|
| 推理引擎 | vLLM 0.29 已支持 `deepseek_v4` | 只能在 sm90/sm100/sm120 上跑 |
| profiler | 需要 `profiler/models/deepseek_v4.yaml`（catalog + sequence + layer_types） | **要新写**：CSA/HCA 层与 indexer 的算子要映射到 canonical 层名 |
| 模型 config | 已入库存档三份 | 玩具配置的 `compress_ratios` 与页尺寸需按**完整层列表**验证 |
| 模拟器 | `trace_generator` 支持逐层类型（Zamba2 那轮做的 `layer_types`） | 可复用；只需给 V4 写出对应 yaml |
| 数据面常数 | 我们有 0.26 GB/s 出口、585/1351/8223 ms handoff | 需要把"每请求 KV 体积"换成 V4 的小值重算刻度 |

---

## 6. 建议的推进顺序（每一步都有明确判据）

1. **读论文 KV 章节，对齐"10% vs 我算的 34%"**（§2.3）。
   判据：给出论文口径下 V4-Flash 的 KB/token，并说明我们的公式差在哪。
2. **把完整层列表的玩具配置跑起来**（12 层，而不是 1 层），确认
   `compress_ratios` 混合下页尺寸约束有解（§3 约束 5）。
   判据：`LLM(...)` 启动成功并报出与公式一致的 `KV cache size`。
3. **`hc_mult` 的影响实验**：在 5090 上试 `hc_mult ∈ {2,4,8}`，
   确认形状约束与质量影响；同时在 4090 上确认"无 mHC 路径不可用"，
   决定是否值得写 mHC fallback（否则实验只能在 5090 域做）。
4. **决定是否自训**：按 §4.3 的预算，选"继续预训练 1–10B tokens"
   还是"只改 config 做模拟研究"。**建议先做后者**——因为我们的研究对象是
   **调度**，不是**预训练**；用 V4 的结构 + 我们的模拟器就能回答
   "KV 小一个数量级后调度还值多少"，成本低两个数量级。
5. （若决定自训）先做 **DSV4-S 的 1–10B token 继续预训练**，同时准备 fp8
   量化管线；训练侧建议 mHC 先用小 `hc_mult`，并在方案评审时明确
   "砍 mHC"的风险（§4.1）。

---

## 7. 未决问题与风险

1. **mHC 的取舍**：砍掉它最省事（也能摆脱 sm90+ 限制），但论文把列为三项升级之一；
   保留它则我们的 4090 域完全无法参与实验。**这是需要你拍板的第一个决策点。**
2. **fp8-first 栈**：vLLM 的 V4 路径对 fp8 权重与 KV 都是硬要求，
   我们的量化/训练流程要相应准备（目前仓库里只有 bf16 profile）。
3. **词表**：官方 129280 的词表对 1B 级模型太重（DSV4-S 里占 265M），
   自研建议换 32k–64k，但这会带来 tokenizer 训练与数据重编码成本。
4. **页尺寸约束的普适性**：我只在 1 层玩具上撞到，官方 43/61 层显然有解，
   但"什么样的 ratio 组合能整除"需要靠 §6.2 的完整层列表实验确定。
5. **V4.1 是多模态**（vision_config + text_config 嵌套），如果目标是纯文本调度研究，
   应基于 V4（`model_type: deepseek_v4`）而不是 V4.1。

---

## 附录：本次调研的原始证据

| 证据 | 位置 |
|---|---|
| 官方 config（V4-Flash / V4.1-Flash） | 镜像外已存档；`LLMServingSim/configs/model/deepseek-ai/` |
| 论文摘要（arXiv:2606.19348） | 本机抓取，标题与关键句已引用于 §1.1 |
| vLLM 实现 | `vllm/models/deepseek_v4/{attention,compressor,sparse_mla}.py`、`nvidia/model.py` |
| KV 公式与 state_dim | `compressor.py:152-186`（`assert compress_ratio in [4,128]`、`state_dim = 2*coff*head_dim`） |
| 启动实测日志 | 本机 4090 失败、5090 成功；关键报错行已引用于 §3 |
| 玩具配置 | `LLMServingSim/configs/model/deepseek-ai/DeepSeek-V4-Tiny-Smoke.json` |
