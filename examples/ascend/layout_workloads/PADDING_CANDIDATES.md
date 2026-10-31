# 现有 kernel 的 padding 候选与准入限制核查

范围：静态检查本仓库 KDA、GDN、Ascend RMSNorm/FP8/Attention/DeepGEMM、
Sparse MLA、K-pool、TopK 和 mHC 的相关实现，以及当前 Ascend 布局优化代码。
这不是设备性能报告，也不是对所有参数和全部 lowering 分支的运行验证。
以下“阻塞因素”区分源码可见的结构与移植/简化后仍需由 IR 确认的条件。

## 结论

### padding 专用检查的更新

新增搜索现在使用独立的 `CanChangeUbPadding`；ownership 的 reducer 排除规则保持。
下文原始候选审计中的 reducer/部分 copy/外层索引限制，需结合以下新范围理解：

- 允许保留已推导的 fragment/loop 布局，支持同 dtype、静态 extent、规则 offset
  的部分 SIMT copy；不重新分配线程。
- 支持外层 serial/unrolled 索引的静态逐次评分，最多 1024 次动态调用；超过上限
  或无法确定地址/边界时回退。这里枚举全部调用，不是假定第零列代表其他列。
- reducer、初始化以及其他未评分操作可作为不变项留在组件中，前提是它们不直接
  访问候选 UB storage。若 reducer 直接读取该 UB，仍拒绝 padding 它。
- 比较指标只覆盖可评分 SIMT 操作；诊断 `cost_scope` 为
  `padding_scored_ops_only_invariant_ops_excluded`，不变操作是排除的共同项，
  不是零成本。完整函数指标仍可能未知。
- 原有 DMA、alias、逃逸、容量、用户 UB 物理布局等保护保留。Fill 直接写候选
  UB、动态长度、复杂控制流和多版本等仍可能阻止原始完整 kernel。

因此 Flash Decoding 的“fragment 归约结果→独立 S_shared 当前列”结构可进入新检查，
但不代表完整原 kernel 已移植或验证。KDA 微基准若用同一个 work 做归约输入和
列写回输出，仍会因 reducer 触及候选 storage 而拒绝，不能误报为全量支持。

新增回归覆盖部分列 copy 的每列实际写入、线程布局不变、1024 相位预算，
以及 reducer 访问/不访问输出 storage 的两种情况；需要编译后运行 TVM 测试。

诊断/检查版本 3 进一步允许非 GM↔UB 搬运作为不变项留在组件，例如 QK 的
L0C→logits UB。候选不能触及这些搬运使用的 storage；并不支持 padding 任意
矩阵搬运目标。`flash_decode_qk_column_store` 将真实 Cube BF16 QK 生产接入列写回，
单缓冲、单 AIV，具体参数/命令见 README；仍未实现完整 attention 或证明性能收益。

目前仍没有确认一个原样可运行、受当前 pass 支持、并已证明 padding 优于
ownership 的真实完整 Ascend kernel。新找到的结构候选不等于已发现收益。

- KDA 的固定列写回有最直接的稀疏地址证据，但默认有效行数很少，必须先核对
  每个 warp 实际有几个写 lane；不能用 32 活跃 lane 的微基准代替原配置。
- KDA 三角求解、GDN GEMM 中间存储存在约束，但不保证访问冲突；可能已采用
  连续访问、broadcast 或适合矩阵指令的布局。
- 原生 Ascend RMSNorm 和 FP8 cast 确实被新增优化的检查限制，但其 UB 访问
  本身偏连续，暂不应为它们优先投入 padding 支持。
- Ascend DeepGEMM 转置是值得单独研究的真实 UB 场景，但它是 SIMD scatter，
  不能用当前 SIMT entry 模型证明或评价收益。

所有 kernel 必要的基础布局推导仍执行。这里的“不进入 pass”特指不进入新增的
ownership/padding 搜索，不是整个布局推导 pass 不运行。

## 候选实现与建议优先级

新增可运行阶段提取例子：`run_workloads.py --case flash_decode_column_store`。
来源为 `examples/flash_decoding/example_gqa_decode_varlen_logits.py` 的逐 tile
`reduce_max → fragment → S_shared[:,k]`，实现见 kernels.py，参数与命令见 README。
归约输入与结果 UB 独立，不强制 fragment 放置；附实际 lowering 地址/mask 审计。
这用于验证候选，不代表已证实 padding 收益或已移植完整 Flash Decoding。

| 实现 | 实际代码/数据流 | 适合 padding 的理由或反证 | 当前阻碍 | 可行性与建议 |
| --- | --- | --- | --- | --- |
| [KDA token-parallel](../../kda/chunk_intra_token_parallel.py) | `reduce_sum(Aqk_shared, Sum_Aqk_fragment)` 后 `copy(..., Sum_Aqk_shared[:, j % CS])`，另一支写 `Sum_Akk_shared[:, d]` | 每个 epoch 的目标是一列，不能像完整逐元素计算那样随意换成同一行的相邻列；但每请求是否有多个活跃写 lane 尚未确认 | CUDA 原实现需移植；归约/初始化 TileOp；部分列 SIMT copy；循环变量改变列索引；同组件 `exp2`；`Beta_shared` 的并行 GM 读取；block 相关迭代数和条件；启用多阶段时还有 DMA 版本限制 | **先核对 lane/mask，后决定是否开发。** 固定 ownership 的部分列物理重映射可分步支持；原形态完整支持涉及多项检查和成本分析，难度中到高 |
| [KDA 三角求解](../../kda/chunk_inter_solve_fused.py) | `Aa_mul[i,j]=a[i]*Ai[i,j]`，`reduce_sum(..., dim=0)`，随后条件更新 `Ai` 的某一行，下一迭代依赖已更新矩阵 | 有真实循环依赖，不能任意融合/跨迭代重排；但矩阵读取可能本来沿 j 连续，归约也可能在寄存器中完成，**尚无冲突证据** | CUDA 移植；reduce、GEMM、fill；`exp2`/条件表达式；可变边界流水线；条件区域；切片 GM 的非连续性（取决于 H/范围）；同一存储还连接其他计算阶段 | **暂缓完整 kernel。** 先测 forward-substitution 的实际 UB 地址；若确有瓶颈，再支持固定 ownership、静态 BC 的一个 epoch；完整循环与 GEMM 契约难度高 |
| [GDN recompute W/U](../../gdn/example_wy_fast.py) | `V_shared*Beta -> U_Beta_shared -> GEMM -> U_fragment -> U_shared -> GM`；W 路径类似 | GEMM fragment 布局可能限制生产者线程分工；但中间值写回可能已连续、已有 swizzle，移植到 Ascend 后还可能变成 L0C→UB DMA | CUDA 移植；GEMM TileOp；K/V 的显式 swizzled layout；`exp`；Beta/G 的直接 GM load；流水线；按 bh 切片的 GM region 可能不连续；输出 fragment→UB 可能有 dtype 转换 | **不作为首个 padding 目标。** 必须先明确 Ascend 上对应的是 SIMT 访问还是 DMA；不能把 CUDA MMA→shared 的冲突直接等同为 Ascend UB 冲突 |
| [Ascend RMSNorm](../example_rmsnorm.py) | UB→x_frag，平方和 reducer，rsqrt，x_frag×rstd×weight→UB；小 RSTD 切片写回 | 原生 Ascend SIMT，线程分配有 reduction/coalesced-width 约束；但 x/weight 读取沿尾轴连续，RSTD 每次只写少量连续值，不是明显稀疏冲突 | reducer；显式 `coalesced_width=1`；`rsqrt`、可能保留的条件/if_then_else；部分区域 SIMT copy；显式 buffer versions；主要 x/y UB 为一维，当前 padding 不产生候选；dtype 非 f32 时还需核对 copy cast | **可实现但收益依据弱。** 放宽固定 fragment/loop 约束对 padding 的限制可行，原 x/y 访问仍未证明需要 padding；先查实际访问，不建议先做完整支持 |
| [Ascend per-token FP8 SIMT](../example_per_token_cast_to_fp8.py) | 固定 y_local float2 fragment，abs/absmax，scale，归一化，转 FP8，写回 UB | 确有固定线程布局，但输入注释及映射明确为连续 float2 strip；b64 的分片开销不一定能被插空洞解决 | 显式 fragment layout；归约；abs 等调用；异 dtype fragment copy；versions/Persistent 上下文；尾部/条件与布局分支需按参数核对 | **低优先级。** 不能把 FP8 dtype 本身当禁用原因；真正限制是转换、约束和可能的部分字写成本。即使全部支持，连续输入也未必受益 |
| [Ascend DeepGEMM scale 转置](../deepgemm/kernels/transform_sf.py) | `_transpose_16rows` 用 `vld + vintlv + vsstb`；converted/padded stride；多种打包和重复输出 | 真实 UB 转置/散写，线程 ownership 不是这段 SIMD 的优化手段，可能值得独立调 pitch | SimdVF；raw UB pointer/access_ptr；大量 `T.view` 字节重解释；不同 dtype alias；动态 group/while/条件；versions；部分 DMA；显式 scatter stride/打包契约 | **独立 SIMD 研究方向，非当前 pass 的小修补。** 先手工试合法 pitch 并测 SIMD 指令行为；需专门布局 rewrite 与成本模型，不能直接套 SIMT key/group/entry |

表中一项检查失败就足以回退；列出后续因素是为了避免“删掉第一个拒绝条件后
仍然不优化”的误判。原 CUDA 文件不会调用 AscendLayoutInference；上述检查
针对按对应算法结构移植到 Ascend、且操作保留到检查点时的预计阻碍。
编译期常量折叠可能消除部分条件，因此具体触发项最终需核对布局推导前 IR。

## KDA 必须先澄清的两个问题

### 实际活跃 lane 数

函数参数默认 `block_H=1`；`get_configs()` 的候选为 1、2、4、8，而不是 32。
归约后的 fragment→shared copy 会受结果放置和 replica-zero store 规则影响。

- 如果一个 warp 对应一个 head、只有一个 lane 写结果，该请求已经只需一个 entry。
  不同 warp 写相同 group 不等于同一个 request 内发生冲突。
- 如果同一 warp 内有多个 head 的结果写入固定列，才需要进一步检查地址分布。
- 原始输出 dtype、CS/SCS、有效 lane mask 都会改变结果，不能统一套 f32/32 lane。

现有实验框架要求 rows 整除 threads，默认至少 32 行；这无法原样复现 1～8 行
的默认 KDA 输出。要评估原配置，应支持部分 warp 并用实际 mask 选周期，而不是
先把 block_H 放大再宣称原 kernel 有收益。

### 归约不天然禁止物理 padding

ownership 可能改变线程分工，必须尊重 reducer 的参与线程和结果放置；padding
只改变 UB 地址，原则上可以固定 reducer/fragment 布局。但现在 padding 复用了
`CanChangeUbOwnership` 的资格集合，因此也被 reducer、显式 fragment/loop 布局等
条件一起挡住。这个限制是当前实现的支持范围，不是硬件“不允许 reducer+padding”。

可行路线是单独建立 padding 合法性检查，保持线程布局、同步、参与者和结果
placement 不变，检查候选 UB 的所有使用者；不是直接删掉 reducer 判断。
对于候选间不变且不触及候选 UB 的归约操作，可以在证明不变后使用共同成本；
归约内部 scratch UB 或访问发生变化时必须展开分析，不能将未知成本设成 0。

## 当前实现的完整检查清单与扩展可行性

以下列出当前准入/候选/打分阶段的独立条件。它们针对关联组件，不是整个函数
一旦出现某种操作就全局禁用。例外是当前 DMA 多版本标志按函数保守收集。

| 检查 | 来源函数/数据 | 当前要求 | 扩展可行性 |
| --- | --- | --- | --- |
| 组件覆盖 | `FindOwnershipEligibleOps`、推导 use-list、同 `buffer->data` 关系 | 整个关联组件都满足资格，不能只放行一个写回操作 | 可为 padding 建单独使用者图；不能漏掉 reducer、copy、view 或 opaque 使用者 |
| 操作类型 | `CanChangeUbOwnership` | SIMT Parallel、受支持 SIMT copy、GM↔UB DMA；其他类型拒绝 | Fill 可优先做完整 lowering；reduce/GEMM/scan 需各自契约，不能统一忽略 |
| SIMT 区域 | `simt_loops_`、`ownership_simt_copies_` | 受支持的 SIMT 执行上下文 | CUDA 移植、SIMD 支持是独立工程；SIMD 不是打开一个开关即可 |
| 复杂控制流 | For/If/While/SBlockRealize visitors、`ownership_excluded_ops_` | 外层 serial/unrolled，边界线程一致；if/while/非恒真 predicate 保守排除 | 先支持可证明 block-uniform 分支；必须保持 barrier 参与一致 |
| 表达式一致性 | `IsOwnershipUniformExpr` | 边界/DMA 参数不含 load/call、只用白名单变量 | 数据依赖的长度/页表需范围证明或运行时 guard，复杂度较高 |
| 元数据 | `CanChangeBufferOwnership` | UB/fragment；静态 shape/stride/offset；dtype/索引维数支持 | 保守扩展到有界符号尺寸可行，但地址范围、容量和评分都要一起扩展 |
| 线程和循环域 | `CanChangeUbOwnership` | 正的常量线程数；Parallel min=0、extent 为正的静态常量；包含 fragment 时逻辑总域需整除线程数 | 部分 warp/尾部需验证 predicate 与 inverse；KDA 原始小 block_H 尤其不能套满 warp 假设 |
| SIMT 直接 GM 访问 | `access` → `CanChangeBufferOwnership` | 当前循环访问只允许受支持的 UB/fragment，GM scope 会失败 | padding 固定 ownership 时可单独证明 GM 地址不变；ownership 仍需考虑 GM 合并访问成本 |
| 显式布局 | `annotated_layout_map_`、loop layout、`coalesced_width` | 当前不改显式 fragment/loop/UB layout | 可固定线程/fragment 布局后搜索 UB padding；显式 UB 物理布局仍应尊重用户约束 |
| alias/view | `buffer_data_to_buffers_` | 不同 Buffer view 共享 data 时保守拒绝 | 统一成底层 byte-address 映射后证明每个 view 一致，尤其跨 dtype，难度高 |
| 未知使用/指针 | `ownership_unknown_uses_` | 推导列表外访问或 handle 使用会阻塞对应存储 | 原始 pointer 算术需要明确读写区间和映射契约；不能只改 BufferLoad/Store |
| reducer | `HasReducerUpdates()`；初始化/finalize 等非 Parallel TileOp | 不进入新增搜索 | padding 可保持原有 reducer plan，但需独立资格和完整使用者分析；中到高难度 |
| 内部循环体 | `PostOrderVisit` | 不支持 call、局部绑定、内部控制流、非 Parallel 内部循环及相关注解 | 纯逐元素调用可逐项放行；读取/写入/同步副作用及索引调用不能统一放行 |
| 写映射 | `access`、`Layout::DetectInjective` | 写入单射，同一循环内被写 buffer 的读写偏移必须一致 | 不同映射需依赖证明；固定所有权的 padding 与重分配线程应使用不同证明条件 |
| 外层迭代索引 | `outer_vars` 检查 | 外层变量不能改变 SIMT 访问索引 | `j%CS` 等规则情况可按有限相位分析；动态/不规则索引需要更强证明 |
| SIMT copy | `CanChangeUbOwnership` copy 分支 | 无注解/特殊 dst_block；至少一端 fragment；同 dtype、同 shape、完整范围 | 固定列/行 region 的重写与评分是 KDA 最直接缺口；先做静态偏移，再做周期偏移 |
| UB↔L1/L0C 等搬运 | 同上 DMA 分支 | 当前只接受 GM↔UB | 需分别支持对应硬件搬运描述和布局约束，不可当成普通 GM DMA |
| DMA 物理布局 | `TryPlanPeriodicPaddingDMA`、`LowerDMACopy` | 完整静态 UB tile；相同 dtype；GM region 连续且可证明不越界；完整周期；UB base/pitch 满足 32B 对齐；lowering 还要求不使用 data_select 尾部填充 | 静态非连续 GM 的二维 descriptor 可扩展；部分 UB/tail 和 cast 是另一个契约，候选检查需与 lowering 保持一致 |
| DMA 多版本 | `padding_dma_versioned_` | 函数中出现多阶段/软件流水线/显式版本标记时，DMA padding 保守回退 | 先收窄成 per-buffer 标记，再支持 materialized ring plane；需核对各版本大小、偏移和容量 |
| 候选形状/周期 | `TryImproveUbPadding` | 已分配、无布局的多维 dense UB；offset=0、无显式 stride；正的整行 lane stride；固定 +32B；周期来自 stride 的 1～32 倍，必须有内部空洞 | 1D/非整行/实际 mask 的候选可扩展；不能假设任何 padding 都能改善分布 |
| 注入性与容量 | `DetectInjective`、`PaddingFootprint` | 映射无重叠；所有收集 UB 的保守占用可算且不超过 216KiB；对齐 32B | 最终 allocator 复用可提高估计精度，但早期上下文与最终版本/临时存储需一致 |
| 访问成本 | `AnalyzeOwnershipEffect`、`CountInvocations`、UB collector | baseline 与候选都可估计；循环次数/地址可求值；不支持的操作或未知 RMW 成本导致未知 | 规则相位可符号化/压缩；block-dependent trip count 可按代表类或保守相位比较；不规则 TopK 需要新模型 |
| 利润判断 | `IsBetterOwnership`、padding tie-break | 只接受模型改善；padding 同成本时选择更小存储 | DMA 分段、指令供给、同步、地址计算、寄存器压力未完整评分，合法不代表实测更快 |

dtype 不是简单的“FP8 一律拒绝”：当前允许 >=8bit 标量访问，padding 候选字节宽
为 1/2/4/8；异 dtype copy、subword partial store 的未知 RMW 成本是不同问题。
同样，GM 的一个 reshape/view 不一定阻塞 UB：需要看是否真正在被修改的 storage
上产生 alias，以及相关使用者是否连到同一组件。

## 不建议误判成 padding 目标的实现

| 实现 | 原因 |
| --- | --- |
| [K-pool Hadamard/absmax](../../kpool/example_glm53_kpool_compress.py) | ownership 被 shuffle/barrier 和 thread binding 约束，但 `exchange[dim ^ 32/64]` 对每个 warp 仍是连续地址；规约的 `dim+stride` 在当前活跃 lane 内也连续。不能仅因“跨 warp”就认定存在需要 padding 的 UB 冲突 |
| [Sparse MLA gather](../../deepseek_v32/sparse_mla_fwd.py) | `KV[Indices[…],d]` 是 GM 不规则读取；写入的是紧凑 `KV_shared[bi,d]`。UB padding 不会把 GM gather 变连续；后续 GEMM 的 shared/UB 问题需单独检查 |
| [TopK](../../topk/example_topk.py) | logits/max/index 主要是 fragment，不是显式 UB 矩阵；不能把 fragment 第 0 列输出当作 UB 稀疏读取 |
| [mHC pre Sinkhorn](../../deepseek_mhc/example_mhc_pre.py) | 行/列归一化的 `cm` 在 fragment 中。行列 reduce 有约束，但不是现成的 UB padding 目标 |
| [GDN cumsum](../../gdn/example_cumsum.py) | compact `G_shared=(1, block_S)`；算法含 scan 约束，却没有直接的多行稀疏地址证据。内部 scan 是否访问 shared 以及如何访问，必须看 lowering |
| 当前完整 tile 的 scale/finalize 微基准 | 已观察到合法 ownership 可以把线程组织改成连续访问；不能继续当成 padding 独有收益的证明 |

## 建议的下一步

1. **先取实际 lane 地址和 mask，再开发支持。** KDA 优先核对原 block_H=1/2/4/8
   的 fragment 结果放置；每请求至多一个活跃写 lane 时应停止这条优化路径。
2. 仅当存在真实多 lane 稀疏写回时，再扩展**固定 ownership 的 padding 专用资格**：
   保留 reducer/fragment 布局，先支持静态部分列 copy 和其所有使用者。
3. 再支持 `j%CS` 的规则相位，以及它的循环次数评分。验证所有相位、完整输出、
   DMA 和同步；随后才考虑多版本和尾部。
4. 继续使用相同 IR 的四开关对照，报告组件拒绝原因、模型指标、最终 UB、整核
   实测时延。现有 diagnostics 未覆盖所有前置拒绝点，空候选表不能替代完整审计。

不建议一次性删除 reducer/条件/copy 限制，或为制造收益放大 block_H、重复计算。
若真实 lane 分布本来良好，应接受“该 kernel 不需要 padding”的结论。

## 对照实现位置

- [layout_inference.cc](../../../src/ascend/transform/layout_inference.cc)：`FindOwnershipEligibleOps`、`CanChangeBufferOwnership`、`CanChangeUbOwnership`、`TryImproveUbPadding`、各控制流 visitor。
- [ascend_mte_plan.h](../../../src/ascend/op/ascend_mte_plan.h)：`TryPlanPeriodicPaddingDMA`。
- [ownership_effect.cc](../../../src/ascend/costmodel/simtvf/analysis/ownership_effect.cc)：`AnalyzeOwnershipEffect`、`CountInvocations` 和实际访问采集。
