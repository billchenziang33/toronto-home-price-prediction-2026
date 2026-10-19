# 二十种方法、三次累计升级：下一轮模型研究计划

> **状态：提案；未开始训练。** 用户确认本项目的“Process Book”指 `summary.md` 与实验账本。本文件只制定下一轮目标，不表示二十种方法已经运行，也不表示三次升级一定能实现。执行时使用 `executing-plans` 按检查点顺序推进，一次只训练一个模型。

**Goal:** 以当前通过验证的 v2 为起点，审查二十种不同机制的方法，争取找到至少三种有边际收益的方法，并形成最多三次累计升级；在固定预算内交付可复现的正、负结果及最终版本决策。

**Architecture:** 保存 `release_candidate/`、`model_candidate_v2/` 和上一轮输出。新实验放入 `experiments/model_optimization_v2/`、`outputs/model_optimization_v2/`；先独立筛选，再按前一版本逐步组合，最后锁定每轨一个方案做一次六个月复核。两轨可采用不同方法。

**Tech Stack:** Windows / PowerShell；复用 `.venv-opt/` 中 Python 3.13.1、LightGBM 4.7.0、CatBoost 1.2.10、scikit-learn 1.7.2、pandas 2.3.3、NumPy 2.2.6。XGBoost 是可选新依赖，安装预算最多十分钟，只接受兼容二进制 wheel。

## 1. 起点、已有证据与需要补强的环节

| 参照 | Off-market MdAPE | List-aware MdAPE | 本轮用途 |
|---|---:|---:|---|
| 官方 starter | 20.8% | 7.3% | 历史背景；不是本轮升级判断基准 |
| 冻结 HGBR v1 | 10.5955753406% | 4.3596470861% | 保存历史累计进步 |
| 当前 v2 / C0 | 9.1733402669% | 4.2356822010% | **新增收益必须优先对照此版本** |

v2 的 MAPE 为 11.9803440483% / 5.3575403746%；每轨六个月共有 10,069 套房。当前 off-market 为 LightGBM preset 6、log-price、权重 1；list-aware 为 LightGBM preset 5、log(sold/list)，与冻结 HGBR 按 0.75/0.25 融合。精确配置在 `outputs/model_optimization_v1/winner_config.json`。

已经确认的事实：

- 上一轮共 66 次真实数据 fit。新方案在六个月相对 v1 的 MdAPE 分别改善 13.42% / 2.84%，两轨各 6/6 月改善；整体方案通过了预先约定的采用门槛。
- 三开发月的初始 LightGBM 优于初始 HGBR；off-market 调参又带来明显开发收益。它们值得作为主路线，但现有资料不能把最终六个月全部增益精确归因于某一个组件。
- List-aware raw log-price 与 ratio target 的开发 MdAPE 为 4.405501% 与 4.399461%，差约 0.006040 个百分点；比例目标单独带来的收益很小。0.75 融合后为 4.388959%。这些是开发选型数字，不能当作三个独立、可靠的机制发现。
- 正确配置后的 off-market L1 为 9.532116%，略差于 raw log-price 的 9.510640%，本轮不原样重跑。CatBoost 的初始 log-price 方案未胜出；新试验只保留不同目标/机制的窄试验。
- Garage/lot-area、旧 community comparables 未获得稳定整体收益；remarks 修正后的区间跨零。宽 one-hot/SVD 路线成本高。没有新增理由时，不把它们重新包装成新的二十项试验。
- 当前 v2 在高价端、Detached / Semi-Detached 等组仍有较大误差。这是事后诊断，不能用真实售价分位作为预测或校准输入。

执行前必须处理的工程风险：

1. **损失函数标签与实际参数一致。** 旧适配器的 `setdefault` 曾让声明的 L1 保留 RMSE。新 runner 在 fit 前核对显式参数，在 fit 后保存实际 objective / booster 参数；不一致就标记 invalid，不入榜。
2. **统一四舍五入次序。** 已核对旧代码：list-aware 六个月评分混合的是已取整 HGBR，导出脚本混合的是原始 HGBR 后再取整。0.25 权重下，中间值差异至多 0.125 元，最终整数预测可能差 1 元；实际影响还需量化。C0 历史分数保持原记录，另存技术审计。新版本的基准重现与导出使用同一条显式规则；这种修正不计作精度突破。
3. **缓存分两层。** 原始学习器缓存绑定数据、特征、学习器与目标变换代码；最终预测缓存还必须绑定父版本、融合、校准、取整、fold split 与依赖版本。不能因 `run.py` 未进入旧学习器指纹，就复用语义已变化的最终预测。
4. **校准与残差只用过去的折外预测。** 禁止用全量训练模型对自身训练集的预测拟合校准器；禁止用当前评估月的标签拟合该月的修正。

## 2. “三次突破”如何计数

同时记录两种数量，不能互相替代：

- **独立有效方法数：** M01–M20 中，有多少方法相对固定 C0 在指定开发折上通过门槛。同一种方法换六个参数仍只算一种机制。
- **累计升级次数：** C0 → C1 → C2 → C3。每次必须相对直接父版本有新增收益，并保存同房预测、配置差异和消融对照。三个方法分别击败旧 v1，不等于完成三次累计升级。

每次开发升级的规则：

1. 两轨各相对父版本降低至少 **1% MdAPE**，在四个开发月中至少 3/4 月变好；单月相对恶化不得超过 5%。
2. 各轨 pooled MAPE 相对恶化不得超过 1%。正确配对 home-bootstrap 的 MdAPE 差值区间上界须小于零；另报月份敏感性。
3. 一次只新增一个有名称的机制/固定特征包。父版本预测就是移除该新增机制的对照。不得靠更换随机种子、取整、挑房屋或删掉难月计数。
4. 若只一轨通过，记为“单轨开发改善”，保留该轨候选，但不计为一次双轨累计升级。

这里的区间是多次筛选后的采用规则和稳定性描述，不是独立显著性检验，也不证明隐藏集一定改善。三次各 1% 相当于累计约 2.97%，不是三个百分点。

最终六个月的最低采用规则仍需同时满足：相对 C0 两轨各至少改善 2%；各轨至少 4/6 月改善；任一月退化 ≤10%；MAPE 退化 ≤1%；正确配对 home-bootstrap 95% 区间上界 <0；权限、复现、导出全部通过。

| 数值目标 | Off-market MdAPE | List-aware MdAPE |
|---|---:|---:|
| 当前 C0 | 9.173340% | 4.235682% |
| 最低双轨升级幅度：各改善 2% | ≤8.989873% | ≤4.150969% |
| 争取目标：各改善 3% | ≤8.898140% | ≤4.108612% |

目标是争取三次升级，**完成本轮并不以凑够三次为条件**。0、1、2、3 次都可以形成完整结论；未达到采用门槛时保留 v2。

## 3. 固定数据与验证协议

- 只用比赛合成 listings，150,000 行，日期到 2026-05；数据 SHA-256 为 `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`。不加真实房源、外部地图、真实成交或隐藏价格。
- 筛选月固定为 **2025-06、2025-11**。补充开发检验月固定为 **2025-07、2025-10**。四个月共同用于开发决策，后两个也是回顾性开发月，不能称作独立确认集。
- 内层校准种子月为 **2025-02、2025-04**，只提供当时真正折外的预测/残差；不是额外评估集。每个外层月只用 cutoff 之前的种子记录。后续可复用已产生且日期更早的折外记录，不能使用日期更晚的记录。
- 已核验月份元数据：二月 train/eval 为 119,979/1,509；四月 123,419/2,077；六月 127,762/2,283；七月 130,045/2,068；十月 135,814/2,250；十一月 138,064/1,867。
- 按日历月切分：训练 `date_sold < month.start_time`，评估该月售出房屋。房屋不是等间隔时间序列，不直接按行号套用默认 TimeSeriesSplit。
- Off-market 禁用 `list_price`；全部预测输入禁用 `sold_price/date_sold/days_on_market/external_avm_estimate/external_avm_estimate_date`；ID 只作配对键。
- 对历史成交聚合/target encoding，训练行也只能读取自身 `date_listed` 前已售出的历史；评估行可用历史截止为 `min(外层 cutoff, 该房 date_listed)`，并使用严格小于。空历史用缺失值与 count=0 表达，不能用全折目标均值偷偷回填。
- 类别词典、缩放、填补、网格原点、截断参数均由训练部分确定。sklearn TargetEncoder 默认回归交叉拟合使用 KFold，不能直接作为本项目的时间安全编码器。
- 2025-12 至 2026-05 六个月只在最终配置锁定后做一次新候选复核。它已多轮查看，始终称为重复使用的回顾性 benchmark。失败后不在这六个月继续试第二、第三套方案。
- 隐藏文件仅在合格方案最终导出时读取特征并生成预测，不用其分布、ID 或任何价格反馈选模型。

## 4. 二十种方法卡：固定首发版本，不开笛卡尔积

大部分方法只替换 v2 的 LightGBM 分支；list-aware 保留原 ratio target 和 0.75 融合，除非方法卡明确改变它。这样一次主要检验一个机制。表中“候选”均是未验证假设。

| ID | 方法与首发规则 | 赛道 | 优先级 / 关键检查 |
|---|---|---|---|
| M01 | 近期成交加权：`age_days=(cutoff-date_sold).days`，`w=2^(-age_days/365)`，权重归一化均值为 1 | 双轨 | 高；相同 C0 参数，单独检验历史样本权重 |
| M02 | 只训练 cutoff 前 24 个月；不同时叠加 M01 | 双轨 | 高；检验较旧市场阶段是否拖累新阶段 |
| M03 | date_listed 的月份 sin/cos 与季度类别，保留现有连续 months | 双轨 | 高；所有字段预测时可见，不用 sale month |
| M04 | 房屋面积信息包：log(area)、是否区间、区间宽度和宽度/中点 | 双轨 | 高；解析低值/高值并检查非正面积 |
| M05 | 规模和房间交互：area/bedrooms、washrooms/bedrooms、parking/area、type×log(area) | 双轨 | 高；零分母给缺失标记，不无声替换 |
| M06 | house_type_name 改为原生类别，替换六个类型指示变量 | 双轨 | 中；信息不变，检验不同拆分表达，不宣称新增数据 |
| M07 | 合成坐标分块：训练原点的 0.02 度网格类别及 type×网格 | 双轨 | 中；只用合成坐标，不接真实地图 |
| M08 | 按时点的 community×type 历史 log(price/area) 均值、count 与向 municipality×type 收缩，伪计数 50 | 双轨 | 高；与旧未收缩 comparables 区别明确，逐行禁止自标签/未来标签 |
| M09 | 历史市场动量：同 type 的过去 90 天与过去 365 天 log(price/area) 均值差、样本量 | 双轨 | 高；只做过去统计，不用未来市场指数 |
| M10 | C0 折外残差的局部 comparable 修正：同类型最近 30 个、过去 365 天，按 n/(n+50) 收缩 | 双轨 | 低/成本高；只用过去 OOF 残差；无邻居修正为 0 并报告覆盖率 |
| M11 | 训练目标换 Huber，`alpha=0.9`，保留各轨 C0 的 log / log-ratio 目标 | 双轨 | 中；明确实际 objective；不原样重跑失败的 off-market L1 |
| M12 | 预测 log(sold_price/area)，再乘有效 area 还原 | Off-market | 中；面积必须预测时可见且为正；这不是外部价格检索 |
| M13 | 直接美元售价、LightGBM `objective=mape` | 双轨 | 中；只能给正价格目标，不能给可能为负/近零的 log-ratio；训练 MAPE 不等于 MdAPE |
| M14 | 全局乘数校准：过去 OOF 的 median(sold/pred)，限制在 [0.95,1.05] | 双轨 | 中/低成本；每个外层月重新用过去拟合，不用该月标签 |
| M15 | 按可见 property type 的收缩乘数校准，向全局收缩伪计数 200；组内至少 100 条 OOF | 双轨 | 中；不能按真实售价分位校准，稀疏组退回已声明的全局因子 |
| M16 | 样条 hedonic Ridge：数值三次样条 5 knots + 稀疏类别 + 线性时间项，Ridge alpha=10、solver=lsqr | 双轨 | 中；提供不同于树的平滑/趋势假设；变换只在训练上拟合 |
| M17 | ExtraTrees：200 trees、min_samples_leaf=20、max_features=0.8、seed=0、4线程 | 双轨 | 中；数值训练中位数填补，稀疏 one-hot；编码≤512列且≤1GB，否则停止该分支 |
| M18 | XGBoost hist：600 trees、depth=6、lr=.05、min_child_weight=5、lambda=3、原生类别 | 双轨 | 低/可选依赖；CPU、4线程、enable_categorical=True，原生模型存 JSON/UBJSON |
| M19 | CatBoost 的 list-aware ratio 目标 + Ordered boosting：600轮、depth=6、lr=.05、l2=3、has_time=True | List-aware | 中；这是不同于旧 log-price 的窄试验；不再展开六组深度搜索 |
| M20 | C0 与一个有互补残差的挑战者在价格空间先固定 0.5/0.5 融合 | 双轨 | 后执行；复用保存预测；C3 才允许在开发期比较五个预设权重 |

补充约束：

- 首发每卡每轨仅一个配置。额外调参不是新方法，必须占后续组合预算并记录。
- 这是一份二十方法清单，不保证全部有可运行条件。每卡必须有 `not_started/running/completed/no_gain/invalid/timed_out/dependency_blocked/budget_skipped` 状态；未运行不能记为“试过且无效”。
- M14/M15 的历史 OOF 必须来自被校准的同一模型版本；若缺失匹配 OOF，先计算依赖成本，不能借用另一个模型的残差凑齐。相关内层 fit 全部计入预算。
- M10 的距离使用训练数据 IQR 标准化后的 latitude/longitude 欧氏距离；其历史残差须能按 ml_num 与训练 listings 的 date_sold 配对，禁止凭 month 猜可用日期。M16 的 knot 取训练数值分位，填补与稀疏编码也只在训练上拟合。
- 直接价格目标若输出非正或非有限价格，标记 invalid 并停止该分支；不能事后截到正数掩盖失败。M20 先从已完成的不同模型族候选中按开发 MdAPE 选一个，再报告与 C0 的残差相关性，不因“算法不同”就预设互补。
- 轻量模型也需独立验证。没有新库、参数更多或模型更新年份，并不构成预测收益证据。
- 禁止把多个随机种子、同一方法的多个参数、普通取整或一个代码修正凑成二十种方法/三次突破。

## 5. 预算与逐阶段操作

总硬上限建议为 **240 分钟墙钟或 180 次真实数据学习器/校准器 fit，先到者停止**。时间包括实现、依赖、特征生成、失败、复核、导出与文档。一次只运行一个模型，最多四线程。下面是可调配的分配，不是训练耗时保证。

| 阶段 | 最多新增 fit | 参考时间 | 必须交付 |
|---|---:|---:|---|
| A0 冻结审计、协议实现与 C0 OOF | 12 | 25分钟 | baseline审计、有效objective检查、fold/OOF/后处理规则 |
| A1 二十卡可行性与两筛选月实验 | 80 | 80分钟 | 每卡状态、完整预测、运行成本和初筛榜 |
| A2 每轨最多四个候选补两开发月 | 16 | 25分钟 | 四个月同房比较、独立方法效果表、C1决策 |
| A3 C2/C3 两次累计组合检验 | 32 | 40分钟 | 父子版本差异、增量指标、是否升级 |
| A4 唯一锁定候选的六个月复核 | 24 | 35分钟 | 同房对照、月度/分组/区间、采用决策 |
| A5 重复、全量拟合、导出与收档 | 8 | 25分钟 | 可加载模型、CSV、复现、哈希与summary |
| 明确失败/重试余量 | 8 | 10分钟 | 失败原因；不拿余量追加新方向 |
| 合计 | **180** | **240分钟** | 达到上限就交付已有结果 |

fit 数按实际模型组件累计，不把一个 ensemble 当一次：C0 的 list-aware 每个新月份需要一个 LightGBM 和一个 HGBR。A0 的二月、四月、七月、十月四个新月份共最多 12 个基准组件 fit；六月/十一月只在指纹和规则匹配时复用上一轮结果。样条/Ridge、残差模型、校准器、每个 seed、内层 early-stopping 后的重新 fit、失败重试都计数。聚合/缩放本身另记构建成本；缓存命中、保存预测的评分和预设融合权重枚举不增加模型 fit，但必须记录选择次数。

复杂方法的内层依赖同样占对应阶段配额；不能按“二十方法×四折”忽略额外训练。阶段配额之间只可预先记录调配，总上限不能增加。新 pipeline 每轨最多三个训练组件。没有足够预算重现完整 pipeline，就减少筛选数量或跳过该方法。

每次开启 fit 前计算：已用预算 + 本次成本 + 最终评估/导出预留。最终预留至少 60 分钟，并按实测耗时×1.3 检查。第 180 分钟后不得开启新方法或新组合。开发 fit 上限 180 秒，最终 fit 300 秒，单卡特征构建上限 120 秒；可恢复记录失败，不默默改算法。

### A0：建立可以相信的比较起点

- [ ] 检查 v1 的 53 项 manifest、v2 manifest、当前数据哈希和上轮 66-fit 账本；保存源文件哈希，冻结本轮 protocol 与方法卡。
- [ ] 独立从保存的 v2 六个月逐房文件重算 C0 MdAPE/MAPE，不重训来“修正”历史分数。
- [ ] 审计 list-aware 后处理次序差异，记录不同预测行数、最大金额差和指标差；选择与公开 C0 benchmark 一致的显式组件取整次序作为本轮 C0 重现规则。新候选沿用同一规则并绑定哈希。必要修正单列，不计精度收益。
- [ ] 为 objective 参数覆盖、未来历史污染、同月校准污染、错配ID、未知类别、样本权重、逆变换、融合取整、缓存语义写聚焦测试。测试用微型构造数据，不计真实 fit。
- [ ] 仅在上述检查通过后生成二月/四月内层 OOF 和七月/十月 C0 对照。缺少或无效缓存不能直接拿全量模型替代。

### A1：先判断二十种方法是否值得留下

- [ ] 实现一张方法卡的固定版本，先核验输入、目标、实际 objective、预测正值和耗时，再到六月/十一月运行。按高、中、低优先级推进；M20 在挑战者预测存在后运行。
- [ ] 保存每个方法每轨每月逐房预测，和 C0 做同房比较。初筛先看两个月合并 MdAPE，再看逐月、MAPE、最差分组、组件数和耗时。
- [ ] 两月都比 C0 差的卡停止，不追加随机种子或宽参数网格；月度混合结果只保留必要的前四名名额。
- [ ] 每轨最多选四个可复现候选进入 A2；若不足四个则少运行。未运行卡给出具体预算/依赖/数据原因。

### A2：选出 C1，区分“低一丁点”与可采用的增量

- [ ] 候选补七月、十月，按四个月 pooled MdAPE 和逐月门槛比较；不得只比较筛选两月。
- [ ] 相对固定 C0 计算方法效果，并检查正确配对区间、房型、预测价分组和月份敏感性。真实售价分位仅放事后诊断。
- [ ] 两轨独立选最优合格方法；若共同达到一轮双轨门槛，记为 C1。否则分别记录单轨改善或无改善。
- [ ] 保存 `checkpoint_C1.json`，包括 parent=C0、每轨方法、精确参数、预测哈希、选择过程和没有通过的门槛。

### A3：争取 C2、C3，而不是重新开始二十项搜索

- [ ] C2 只在 C1 上添加一个有互补机制且已通过独立开发检验的方法；每轨最多三个固定组合。新的组合与其直接父版本对照，不再次与弱 v1 对照计数。
- [ ] 两筛选月初筛后，只有胜出组合补另外两开发月。父版本的同月预测直接复用；这就是新增机制的消融对照。
- [ ] C3 优先检查过去 OOF 校准或一个互补挑战者的简单融合。若校准规则不适用或残差高度相关，明确跳过，不为凑第三次升级强加模块。
- [ ] 融合最多 C0/当前主模型与一个挑战者，价格空间 `w∈{0,.25,.5,.75,1}`；只在开发期选权重。权重与模型全部锁定后不能按六个月结果改动。
- [ ] 每次未过门槛则保留父版本，写 `promotion=false`；不把“计划做了三轮”写成“实现了三次性能突破”。

### A4：最终一次复核

- [ ] 冻结每轨唯一最终 pipeline、权重、特征、目标、历史状态构建规则、校准训练期、后处理和代码/环境哈希，生成 `freeze_manifest.json`。
- [ ] 对 2025-12 至 2026-05 运行该唯一 pipeline；所有月份训练与校准严格遵守各自 cutoff。C0 与 v1 的保存预测只作配对基准。
- [ ] 同时报相对 v1、相对 C0、开发父版本增量；最终采用门槛只对 C0 判断，不能用对 v1 的大降幅掩盖本轮没有新增收益。
- [ ] 正确统计量为 `median(APE_new)-median(APE_ref)`；在同一房屋索引上重抽样重新算两个中位数。另报月份成组敏感性，明确只有六个群组与多轮选型的限制。
- [ ] 如果最终六个月未过门槛，保留 C0；不把下一名候选拿来继续查看相同六个月。

### A5：复现、导出、结束

- [ ] 对每轨一个固定代表月/固定种子重跑，比较逐房价格和指标；序列化加载后预测一致。调换随机种子追分不允许。
- [ ] 两轨均通过最终规则才导出独立 `model_candidate_v3/`；只一轨通过则保存研究结果，保留完整双轨 v2 提交候选。
- [ ] 全量训练与内层 OOF 重建也计预算；如果余量不足则不导出未验证模型，结论标记“研究改善、导出受限”。
- [ ] CSV 必须为 `ml_num,predicted_price`，与各自测试行数/ID/顺序匹配，价格有限且正。模型加载与后处理和评分入口使用同一实现。
- [ ] 写正负结果、二十卡状态表、三检查点父子证据、预算与失败；更新 summary 与桌面归档后停止。本轮不改报告/slides、不等待视频、不发布或提交表单。

## 6. 新实验的文件边界、日志与验证入口

旧 `experiments/model_optimization_v1/run.py` 已超过千行，不再把新路线全部追加进去。保留旧包可重现，下一轮按职责建以下小模块：

| 路径 | 职责 |
|---|---|
| `experiments/model_optimization_v2/protocol.json` | 日期、父基准、哈希、预算、门槛、后处理规则 |
| `experiments/model_optimization_v2/method_registry.json` | M01–M20、适用轨、固定参数、依赖、状态 |
| `experiments/model_optimization_v2/folds.py` | 日历 cutoff、隐藏答案、内层 OOF 可用时点 |
| `experiments/model_optimization_v2/features.py` | 当前冻结特征与各个新增特征包、as-of 历史 |
| `experiments/model_optimization_v2/learners.py` | 学习器、样本权重、实际 objective 断言 |
| `experiments/model_optimization_v2/pipeline.py` | 父子 pipeline、校准、融合与唯一取整规则 |
| `experiments/model_optimization_v2/evaluation.py` | 配对指标、正确 bootstrap、月度与分组诊断 |
| `experiments/model_optimization_v2/artifacts.py` | 两层缓存、原子日志、模型导出、manifest |
| `experiments/model_optimization_v2/runner.py` | 阶段调度、超时、fit 账本、恢复与预算预留 |
| `tests/test_model_research_v2.py` | 上述泄漏、损失、缓存、后处理和复现契约 |
| `outputs/model_optimization_v2/` | 方法状态、trial ledger、OOF、三checkpoint、最终对照与失败 |
| `model_candidate_v3/` | 仅在最终双轨验收后生成的新提交候选 |

每次 fit 前后原子保存/flush：method_id、parent_id、track、fold、数据/代码/配置/依赖哈希、训练/预测ID哈希、目标、声明与实际objective、随机种子、线程、训练权重规则、OOF最大可用日期、开始结束时间、耗时、状态、异常路径和预测哈希。算法级与最终 pipeline 的缓存指纹分开，不凭文件名续跑。

每个方法卡的结论必须分为“代码/日志确认”“合理假设”“尚未验证”；方法在某个赛道有效不自动推广到另一个赛道。

执行前已经存在的核验命令：

```powershell
$env:TEMP=(Resolve-Path tmp).Path
$env:TMP=$env:TEMP
& '.venv-opt\Scripts\python.exe' release_candidate\tools\verify_candidate.py --source-repo .
& '.venv-opt\Scripts\python.exe' -m pytest -q tests --basetemp=.pytest_cache\research-v2-basetemp
```

下一轮计划创建的 CLI 入口如下，**当前尚不存在，必须先实现并通过契约检查才能运行**：

```powershell
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner preflight
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner screen --resume
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner confirm --resume
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner compose --resume
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner freeze
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner benchmark --resume
& '.venv-opt\Scripts\python.exe' -m experiments.model_optimization_v2.runner package
```

`--resume` 只续同一冻结配置未完成的折，不允许在已查看最终分数后换配置再跑 benchmark。预算耗尽时 runner 必须生成部分结果报告并退出，而不是持续挂起。

## 7. 方法资料与证据边界

以下官方资料证明接口/方法可用，不证明在本比赛能降低误差。具体参数是本计划的预设研究选择：

- [LightGBM LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)：训练样本权重与实际 objective 属性；用于 M01 和防止损失配置错位。
- [LightGBM 参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)：Huber、MAPE 等候选训练目标；MAPE 与本比赛 MdAPE 需要区分。
- [scikit-learn 时间特征示例](https://scikit-learn.org/1.7/auto_examples/applications/plot_cyclical_feature_engineering.html)：周期特征、样条与时间验证；用于 M03/M16 的实现参考。
- [scikit-learn ExtraTreesRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)：M17 的另一类树集成。
- [XGBoost 原生类别](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)：M18 使用 hist、类别 dtype 与 JSON/UBJSON 模型保存。
- [CatBoost 通用参数](https://catboost.ai/docs/en/references/training-parameters/common)：M19 的 Ordered boosting；可能更慢，不能据此预设会优于当前模型。
- [sklearn TargetEncoder](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.TargetEncoder.html) 与 [TimeSeriesSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)：默认 KFold/按行切分不能代替本项目的逐日历月、逐时点历史协议。

本轮不采用深度表格大模型、LLM 文本生成、真实市场价格检索或无限 AutoML。这些可以是另一个研究问题，但当前预算优先用于可解释、可审计且能公平对照的二十个机制。

## 8. 本次计划交付自检

- [x] 已读取比赛 summary、实验账本/报告、官方 handout 和 rubric；用户确认没有另一份 SSU Process Book 需要补读。
- [x] 明确 C0 为当前 v2，旧 v1 和 starter 只作历史参照。
- [x] 二十方法与三次累计升级分别定义；没有把不同参数当不同方法。
- [x] 时间/fit 预算可相加，内层组件和失败计数，最终留有预留。
- [x] 四个开发月、校准种子月、六个月回顾性 benchmark 的角色分清。
- [x] 损失参数覆盖、OOF校准、取整次序、缓存指纹的已知风险进入前置审计。
- [x] 成功、部分改善、无收益、运行受限均有停止和交付标准。
- [x] 仅生成计划与目标文字，没有启动新训练、创建新活动目标或外部提交。
