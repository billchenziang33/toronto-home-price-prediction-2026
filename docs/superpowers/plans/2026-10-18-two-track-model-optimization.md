# 两赛道模型与预测方法优化计划

> **执行状态：已完成（2026-10-18）。** 以下勾选记录实际执行；详细指标与交接见 `outputs/model_optimization_v1/comparison.md` 和 `summary.md`。

**Goal:** 在有限预算内，比较模型算法、参数和预测目标，寻找同时降低 off-market 与 list-aware 时间前推验证 MdAPE 的可复现方案，交付完整对照证据；没有可靠收益时保留原候选并报告负结果。

**Architecture:** 保留 `release_candidate/` 为冻结参照，在独立实验目录实现共享的时序评估器和模型适配器。两条赛道独立选算法、参数和目标变换。先用较早月份开发，再锁定方案，对既有六个月基准做一次完整复核；仅合格方案生成独立的 `model_candidate_v2/`。

**Tech Stack:** Windows / PowerShell；Python 3.13.1、scikit-learn 1.7.2、pandas 2.3.3、NumPy 2.2.6、pyarrow 22.0.0。CatBoost 1.2.10 与 LightGBM 4.7.0 安装并验证于隔离的 `.venv-opt/`；XGBoost 与 Optuna 未安装，也未用于本轮。

## 状态、目标与预算

- 状态：**已完成**。实际从 2026-10-18 20:47:29 UTC 执行至约 21:28 UTC；累计 66/72 次真实数据模型 fit，低于 180 分钟与 72 次 fit 上限。
- 冻结参照：commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`，代码与 CSV 在 `release_candidate/`。MdAPE：10.60% / 4.36%；MAPE：13.40% / 5.57%。最终比较重新从冻结逐房预测算未四舍五入值。
- **争取目标：两轨 MdAPE 各相对下降至少 5%**，按公布基准约为 off-market ≤10.07%、list-aware ≤4.14%。这是研究目标，不是保证，也不是无限运行条件。
- **最低升级门槛：两轨各相对下降至少 2%**（约 ≤10.39% / ≤4.27%），并满足下文稳定性、MAPE 和可复现性要求。5% 是争取目标；2% 是本轮预先约定的最低采用幅度，二者均属于团队决策规则，不是比赛官方评分线。
- **单轮硬上限：180 分钟墙钟时间或 72 次真实数据模型 `.fit()`，先到者生效。** 包括环境准备、失败、重试、最终复核、导出及写记录。一个算法/参数/赛道/月份/种子的真实数据 `.fit()` 算一次；内部另行重拟合也计数。构造的微型单元测试不属于搜索试验。
- 各阶段分钟数是可相互调配的上限建议，不是训练耗时承诺；总墙钟和 fit 硬上限不能随调配增加。
- 90 分钟时审计剩余预算；最迟第 125 分钟停止开启新方向。随 pilot 耗时估计，为最终评估/导出预留 `max(40 分钟, 预计最终成本×1.3)`；预算不足就缩减搜索，不压缩最终核验。
- CPU 为默认，最多 4 线程，一次只训练一个模型；不假定 GPU 可用。pilot 单 fit 上限 90 秒，开发 fit 180 秒，最终 fit 300 秒。达到超时就记录失败和已完成结果，不能默默换模型或使用未完成分数。
- 早于预算完成就交付；无收益、部分收益或依赖受限也必须形成完整阶段结论。记录视频、等待反馈、发布、表单均不构成本阶段持续运行条件。

## 已有经验与不再重复的工作

1. 已有多个特征实验和一套可用候选；当前模型还没有系统参数搜索或 CatBoost/LightGBM 算法对照，所以先比较算法和小参数集，不能声称现有算法已经最优。
2. 冻结模型是 log(sold_price) 上的 HistGradientBoostingRegressor。Off-market 有 16 个输入：7 个数值字段、6 个房屋类型指示变量、3 个原生类别字段；list-aware 再加 list_price，共 17 个。房屋类型当前是指示变量，house_style / municipality / community 才是原生类别，按代码解释。
3. Garage/lot-area、community comparables、宽 one-hot 和 remarks 已尝试过；本轮不重新开展同样的宽搜索或 SVD 文本方案。Remarks 的小幅点估计改善未通过修正后的区间审计，保留作存档。
4. 旧类别增益无法按原声明复现，不把它当作已验证先验。新实验必须保留精确参数、输入、版本和逐房预测，避免只有报告数字而找不到对应模型。
5. 此前 `median(APE_new - APE_base)` 与 `median(APE_new) - median(APE_base)` 混用。本轮所有重采样都在相同房屋索引上重新计算后者，负值代表新模型更好。
6. **2025-12 至 2026-05 六个月已经被多次查看和选型使用。** 不能再把其中后三个月称为从未触碰的确认集；本轮最终六个月是重复使用的回顾性基准。最终独立成绩仍只有主办方隐藏集。

## 数据与验证协议

- 数据只用给定的合成 listings。已核验 150,000 行，date_sold 为 2021-01-01 至 2026-05-31。
- 数据快照：`96a258439e2a85f650699be8339e52d8ac5614b7`；listings.parquet SHA-256：`18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`。执行前重新比对；不符则明确记录差异并停止把旧结果当成同数据比较。
- **开发月份预先固定为 2025-06、2025-09、2025-11**，分别有 2,283、2,035、1,867 套房。调参筛选只用 2025-06 与 2025-11，最终开发一致性再补 2025-09。它们是新的回顾性开发窗口，不宣称是独立外部测试。
- 每月使用与官方 starter 一致的边界：训练 `date_sold < month.start_time`，评估该月售出的房屋。记录每个 cutoff、训练行数、预测行数和 ID 哈希。特征、词典、编码、统计量只能在该折训练数据上拟合。
- 最终基准保持 2025-12 至 2026-05、每轨 10,069 套，逐房与冻结预测精确配对。选择好算法、目标、参数、融合权重后锁定配置，再对每轨一个候选做一次六个月完整评估；不看结果后继续对这六个月追分。
- Off-market 禁用 list_price，所有输入禁用 sold_price、date_sold、days_on_market、external_avm_estimate 和 external_avm_estimate_date；目标列只进入 fit/评分。ID 不作为预测特征。
- 默认复用冻结 16/17 列特征含义。类别适配差异要记录，初始算法比较不同时添加新内容特征。未知类别按训练词典或算法规定处理，禁止按评估标签生成编码。
- 无外部市场数据、无真实房屋记录、无隐含价格检索。隐藏测试只用于最终合格模型输出预测，不参与调参、融合选权重或选择目标函数。

## 推荐路线与备选路线

| 路线 | 作用与取舍 | 本轮选择 |
|---|---|---|
| 只调当前 HGBR | 环境成本最低，但无法回答换算法是否有收益 | 留作参照和候选，不作为唯一方向 |
| HGBR + CatBoost + LightGBM，小范围调参与目标变换 | 覆盖不同树构建/类别处理，复用现有结构化输入，计算可控 | **主路线**；两个赛道各选赢家 |
| 广泛 AutoML、深度表格模型、LLM 文本、地图和外部数据 | 依赖与搜索空间大，难在本轮形成可靠复核 | 留给另一个独立阶段，不在预算内自动展开 |

CatBoost 与 LightGBM 的原生类别处理、回归损失和调参能力有官方文档支持；这只支持“可开展实验”，不能推断会比当前模型好。XGBoost 作为一个主要库无法安装/运行时的替代，不能变成额外第四轮大搜索。安装总预算最多 10 分钟；只用匹配当前 Python 的二进制 wheel，不现场编译、不改系统 Python。失败要记原因，保留能运行的路线。

### 候选方法与固定搜索空间

1. **直接预测 log(sold_price)**：初始比较三个算法，RMSE/L2 作为统一起点。当前 HGBR 原样参照；CatBoost 首发 `(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=3)`；LightGBM 首发 `(n_estimators=600, num_leaves=31, learning_rate=0.05, min_child_samples=30, reg_lambda=1)`。固定随机种子 0，CPU 4 线程；CatBoost 原生类别按日期、ID 稳定排序并使用 has_time；LightGBM CPU 设置 deterministic=True、force_col_wise=True。库版本和所有生效参数落盘。
2. **仅调各赛道胜出的算法族**：每轨最多 6 个预置配置，不做笛卡尔积；以两个开发月份筛选。未胜出算法不再单独展开搜索。

| 算法 | 六个预置配置，每个括号均按表头参数顺序 |
|---|---|
| HGBR：max_iter / learning_rate / max_leaf_nodes / min_samples_leaf / l2_regularization | (300,.05,31,20,0); (600,.05,31,40,1); (600,.03,63,40,1); (600,.05,15,40,2); (600,.05,31,80,5); (800,.03,31,40,5) |
| CatBoost：iterations / learning_rate / depth / l2_leaf_reg | (600,.05,6,3); (1000,.03,6,3); (600,.05,8,5); (800,.05,5,5); (800,.03,7,10); (1000,.05,6,10) |
| LightGBM：n_estimators / learning_rate / num_leaves / min_child_samples / reg_lambda | (600,.05,31,30,1); (1000,.03,31,50,1); (600,.05,63,50,3); (800,.05,15,30,3); (800,.03,63,100,5); (1000,.05,31,100,5) |

HGBR 新配置使用 early_stopping=False 和固定轮数。新 CatBoost/LightGBM 配置也用固定轮数，不以外层评估月标签触发 early stopping。若确有必要采用 early stopping，只能用 cutoff 之前的内层按时间留出数据，所有额外 fit 计入预算；不得据此额外扩增搜索。XGBoost 若替补，只用一套 `(tree_method=hist,n_estimators=600,max_depth=6,learning_rate=.05,min_child_weight=5,reg_lambda=3)` 初始配置，筛选失败就结束该分支，不额外开六配置搜索。

3. **Off-market 的稳健损失试验**：固定胜者参数，仅把 log 价格的 RMSE/L2 换成 MAE/L1；HGBR 对应 absolute_error，CatBoost 对应 MAE，LightGBM 对应 regression_l1。它是训练替代目标，不等于直接最小化 MdAPE，仍以逐房百分比误差评价。未能验证兼容的损失组合要记录并跳过。
4. **List-aware 的预测目标试验**：固定胜者参数，学习 `z = log(sold_price/list_price)`，预测 `price = list_price * exp(z_hat)`。它把任务表述为“相对要价的溢价或折价”，而非直接重建全部价格。只在训练中计算 sold/list，预测时仅需 list_price；必须检查训练和预测所需价格为正数，不以任意常数默默替换非法输入。该变换只给 list-aware。
5. **低成本融合**：每轨只允许冻结模型与一个新候选相融合，价格空间权重 `w ∈ {0,.25,.5,.75,1}`，`prediction=(1-w)*frozen+w*challenger`。用同一房屋的开发折外预测选权重并锁定；最终六个月直接复用已保存的冻结预测，因此每折只需训练新候选。多层 stacking 和两个新算法的额外融合不在本轮范围内。
6. 价格两端误差较高可作为诊断方向，但本轮不按测试房屋的真实 sold_price 分组或校准。真实售价分位只能用于事后误差图；任何后续校准必须只依赖预测价/预测时可见字段，并在内部时间折外数据上拟合。

## 文件与执行接口

原仓库根目录包含未发布 remarks 修改。所有实验默认显式调用冻结候选代码，不能通过当前根目录 features.py 偷带入 remarks，也不能覆盖根目录或 release_candidate 的 submissions。

| 新文件 / 目录 | 职责 |
|---|---|
| `experiments/model_optimization_v1/configs.json` | 固定月份、候选配置、种子、目标、预算和采用规则 |
| `experiments/model_optimization_v1/adapters.py` | 同一 fit/predict 接口、训练折内编码、log/ratio 逆变换、序列化 |
| `experiments/model_optimization_v1/run.py` | audit / pilot / screen / tune / methods / benchmark / package / all 阶段，进度、超时、缓存、预算 |
| `experiments/model_optimization_v1/metrics.py` | 配对指标、月度/价格/房型诊断、正确的配对 bootstrap |
| `tests/test_model_optimization.py` | 截止日、特征权限、训练词典、逆变换、配对统计量、缓存失效及模型序列化等关键行为 |
| `outputs/model_optimization_v1/` | 所有配置、环境、哈希、逐折日志、预测、分数、失败原因、最优配置、模型和最终对照 Markdown |
| `model_candidate_v2/` | 仅在升级门槛通过时，独立的模型代码/精确依赖/两轨模型文件/预处理/CSV/README/哈希；不混入旧报告 |

执行入口完成后使用 `python -m experiments.model_optimization_v1.run <stage>`；`all` 仅顺序执行有预算且前置通过的阶段。`--resume` 只跳过配置/输入/代码指纹完全相同且已有完整输出的 fit，不能仅凭文件名跳过。

每条预测至少包含 `ml_num,track,month,model_id,sold_price,predicted_price,baseline_price`；同一房屋配对键为 `ml_num,track,month`。目标从输入矩阵剥离后才调 predict。最终表必须显示开发成绩和六个月基准成绩的角色差异。

每次 fit 之前和之后写入 JSONL 并 flush：trial_id、数据/代码/配置哈希、track、cutoff、row counts、seed、线程、开始/结束时间、耗时、status、错误路径、预测文件和指标。超时、异常、依赖缺失、无收益都保留，禁止只保留最好几次。模型与预测用临时文件完成后再原子替换。

## 逐阶段推进

### 阶段 1：基准审计与实验封装（预算 10 分钟；0 fit）

- [x] 验证 release_candidate 的 SHA256SUMS、Git 源提交和两个 CSV；重算未取整基准，不重训已验证基准。
- [x] 固定数据哈希、开发/基准月份、规则及配置。记录系统/库版本，检测缺失、目标正值和两轨列权限。
- [x] 实现最小共享 runner/adapter/metrics，验证 cutoff、off-market 列权限、销售后字段隔离、未知类别和行序。
- [x] 用构造数据验证正确配对统计量（中位数差 -98，不是逐房差值中位数 1），并检查行配对/重复 ID、目标逆变换和缓存指纹失效。
- [x] 验收：`baseline.json`、`configs.json` 与审计 JSON 已生成，16 项聚焦测试通过；冻结包与原始预测未改。

### 阶段 2：依赖与耗时 pilot（预算 15 分钟；最多 4 fit）

- [x] 在 .venv-opt 中验证兼容 wheel 和 import，安装总时长上限 10 分钟；记录失败后使用剩余可运行候选。
- [x] 对一个开发月分别小轮数试跑 HGBR、CatBoost、LightGBM，另一次 fit 检查 list-aware 或目标变换；这四次均计入预算。使用真实训练截止日数据，记录拟合耗时，不把 pilot 小样本/少轮数成绩当正式结果。
- [x] 固定后续线程和计算参数；估计一次完整六个月新候选+全量拟合的时间，重新计算最终预留预算。
- [x] 验收：明确可运行的算法、单 fit 耗时和保留下来的搜索数量。若预算只够窄试验，自动缩减配置而非突破上限。

### 阶段 3：模型族比较（预算 35 分钟；最多 18 fit）

- [x] HGBR 原参照、CatBoost、LightGBM 各一个初始配置，对 3 个开发月、2 条赛道比较；不可用算法可由 XGBoost 替位，总算法数不超过 3。
- [x] 保存所有折外预测，按每轨开发总 MdAPE 排序，同时看月度 MdAPE、MAPE、最差月份和运行时间。
- [x] 两轨分别选一个算法族进入下一阶段，允许使用不同算法。不得凭模型“新旧”或单月最好分决定。
- [x] 验收：开发 leaderboard 与逐房预测齐全，包含缺失/失败/超时条目；未触碰隐藏成交价。

### 阶段 4：胜者小范围调参（预算 45 分钟；最多 24 fit）

- [x] 各赛道胜出的主算法只运行表中的 6 个配置，在 2025-06 / 2025-11 两折比较；已精确匹配的缓存直接复用。若替补 XGBoost 胜出，则该轨跳过六配置搜索，只保留其已验证初参，把预算留给方法比较与复核。先简单、后复杂，差或超时的分支提前结束。
- [x] 若一个候选在两个筛选月份均比当前该轨最优差，且已有完整预测，淘汰而不增加随机种子重试。
- [x] 保存每轨一个参数赢家，明确这是开发选型结果；不因为追逐 5% 目标扩大为网格或无限 Optuna。
- [x] 验收：每轨一个确定配置，整个试验次数与剩余预算可核对。

### 阶段 5：目标函数与融合（预算 20 分钟；最多 8 fit）

- [x] 用各轨调参赢家补 2025-09 验证（最多 2 fit），使赢家具备三个开发月的同房预测。
- [x] Off-market 只测试一个 L1/MAE 版本；list-aware 只测试一个 log(sold/list) 版本，各 3 开发折（最多 6 fit）。
- [x] 复用开发预测比较冻结模型+一个新候选的五个权重，不额外 fit；不按六个月基准结果重新选权重。
- [x] 锁定每轨最终算法、特征、目标、参数、融合权重和预处理。若新方法没有开发收益则保留该轨此前赢家，不重新发散。
- [x] 验收：winner_config.json、开发对照、锁定时间和配置哈希已保存。

### 阶段 6：一次最终复核、导出和交接（预算 55 分钟；最多 18 fit）

- [x] 对锁定的每轨一个新候选跑官方六个月，共最多 12 fit；基准预测直接复用并逐房配对。此步骤之后不再调参追分。
- [x] 生成 pooled/月度 MdAPE、MAPE、signed error、房型/预测价格分组和真实价格分位事后诊断。重算正确配对房屋 bootstrap 及按月份成组的敏感性区间。只有六个月，月份区间稳定性有限；多轮选型后这些都不能冒充独立显著性检验。
- [x] 逐轨应用下文升级门槛；两轨都通过时才生成双轨升级候选。只一轨通过就报告部分进展并保留结果，不声称双轨目标已达成；都没有通过就保留原候选，交付完整负结果。
- [x] 合格时全量训练并导出两个模型；如融合需要重现原基准则最多 4 fit，不能超额。保存模型、预处理、输入 schema、版本与种子。
- [x] 用同一固定种子重做一个代表月/每轨（最多 2 fit），检查指标与预测稳定性；序列化往返须产生相同预测。不能用挑选新种子来改善分数。
- [x] 新 CSV 必须通过官方 check，ID/顺序/行数正确，预测有限且为正；保存候选 SHA256，验证可移植的加载/预测说明，不使用仅在本机存在的绝对源路径。
- [x] 写 comparison.md 和 summary.md：每个方向试过什么、耗时、失败、结果、是否过门槛、剩余不确定性；保存检查点和桌面副本后结束本阶段。

## 升级门槛与结论用语

必须同时满足：

1. 两轨在相同六个月、相同房屋上均至少降低 2% 相对 MdAPE；争取达到 5%。取未四舍五入数字判断。
2. 每轨至少 4/6 月份 MdAPE 较低；任一月相对恶化不得超过 10%。
3. 每轨 pooled MAPE 相对恶化不得超过 1%，防止只优化中位数却扩大严重错价。
4. 每轨正确配对的房屋 bootstrap 95% MdAPE 差值区间上界小于 0；同时展示月份成组的敏感性结果。该门槛是保守采用规则，不保证隐藏期收益；也不能消除重复选型偏差。
5. 数据权限、训练截止日、逐房配对、模型序列化、关键测试、重复预测与 CSV 检查全部通过，结果绑定确切代码和配置。

不满足时输出“探索性改善 / 单轨改善 / 无稳定改善 / 运行受限”之一，并列原因。不得把较低训练损失、单月最好分、其他数据集成绩或模型更新年份当作竞争性能证据。

## 停止与交付

- 达到预算、没有值得继续的候选、或最终复核结束就停止；不得等待视频/依赖/用户回复而把模型目标持续挂起。
- 日志即时保存，阶段后更新 summary；同一指纹成功的计算不重复，文稿修改不触发训练。
- 交付精确基准/新成绩、差值、逐月与分组图、完整试验账本、环境/代码/数据哈希、失败记录、运行耗时、赢家配置。合格时另交可加载模型和两个 CSV。
- 本阶段不生成新 PDF 或重做 slides，不推 GitHub、不提交表单。只在确定新模型是否值得采用后，再独立同步报告和录制材料，避免每次试验重做全部文稿。
- 发布候选 release_candidate 始终保留；只有团队审核后再替换提交版本。计划写好不等于本轮优化已经运行，期望值不写成已达成分数。

## 官方资料（2026-10-18 核对）

- [scikit-learn 1.7 HGBR](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)：现有估计器与可调参数、损失接口。
- [CatBoost 参数调优](https://catboost.ai/docs/en/concepts/parameter-tuning)：类别处理、轮数、深度、正则和 has_time。
- [CatBoost 回归目标](https://catboost.ai/docs/en/concepts/loss-functions-regression)：RMSE、MAE 等；训练目标与 MdAPE 评分必须区分。
- [LightGBM 调参](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)：树复杂度、轮数、过拟合与运行时间取舍。
- [LightGBM 参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)：L1/L2、CPU 确定性与线程参数。
- [XGBoost 参数](https://xgboost.readthedocs.io/en/stable/parameter.html)：可选替代路线，不代表已安装或已有本项目成绩。

## 计划自检

- [x] 两轨分别选模型，替代算法、目标变换、融合均有明确位置；不把当前算法作为硬限制。
- [x] 已有六个月被使用的事实写清；没有虚构未触碰确认集或隐藏集成绩。
- [x] 72 fit 上限：pilot 4 + 比较 18 + 调参 24 + 方法 8 + 最终 18；失败/复现也计入。
- [x] 原候选隔离、现有 remarks 不混入、官方 cutoff 与两轨列权限明确。
- [x] 不要求一定取得收益才停止；不把文稿/录制/提交并入训练目标。
- [x] 研究目标、采用门槛和未运行状态分开，交付和停止条件可检查。
## 实际执行结果与采用决策（2026-10-18）

- CatBoost 1.2.10、LightGBM 4.7.0 已装入隔离环境 `.venv-opt/`；原 `.venv` 未改。真实数据 pilot 4 次完成。每折模型顺序运行，最多 4 线程。
- 18 个算法筛选配置中 4 个精确复用 pilot，另完成 14 次真实 fit。LightGBM 在两轨的三开发月 pooled MdAPE 分别为 10.0368% 和 4.4479%；HGBR 为 10.8787% 和 4.5358%；CatBoost 为 11.1356% 和 4.5250%。
- LightGBM 每轨 6 组固定调参配置在六月与十一月筛选，20 个新 fit、4 个缓存复用。赢家为 off-market preset 6 与 list-aware preset 5，参数详见 `outputs/model_optimization_v1/winner_config.json`。
- 方法比较共 11 个 fit。list-aware 的 `log(sold_price/list_price)` 配合 0.75 新模型权重，在三开发月 pooled MdAPE 为 4.3890%；off-market 保留 raw log-price 和 1.00 新模型权重，pooled MdAPE 为 9.5107%。L1 目标的第一组三个 LightGBM fit 因显式配置仍为 RMSE 而排除；纠正 objective 后的三次 `regression_l1` 试验完整入账，MdAPE 9.5321%，未优于 raw log-price。此修正使用未消耗的筛选阶段 fit 配额，不突破总预算。
- 六个月复核相对冻结模型：off-market 10.5956% → 9.1733% MdAPE、13.4001% → 11.9803% MAPE；list-aware 4.3596% → 4.2357% MdAPE、5.5651% → 5.3575% MAPE。两轨各 6/6 月变好，home-paired 95% 区间分别为 [-1.6102, -1.2453] 与 [-0.1937, -0.0601] 个百分点，两轨均过预定升级门槛。
- 最终阶段实际 17/18 fit：12 个逐月 benchmark fit、3 个全量导出 fit、2 个固定种子重复 fit。两个重复月输出逐房完全一致。新候选在 `model_candidate_v2/`；两份 CSV 分别有 3,974 与 4,064 行，均通过官方 schema、ID、正有限价格检查。模型磁盘往返预测与内存模型一致，manifest 已生成。
- 16 项聚焦测试及完整项目 42 项测试通过；复核输出为 `outputs/model_optimization_v1/comparison.md`、`benchmark.json`、`reproducibility.json`。六个月验证期曾用于先前分析，因此仍属回顾性评估；不能称作独立确认，也不能据此保证隐藏测试成绩。
- 三个最初错误目标标签的 L1 fit ID `0c7914041c5e73c3`、`9c3de7c969210a3a`、`f6ab9b2c6a343f69` 保留在完整账本并在 `winner_config.json` 明确排除。其余失败/缓存/成功状态没有从日志中删除。
- 当前 release candidate 保持冻结不变。接下来须把 `model_candidate_v2` 的结果一致地反映到四页报告、六张幻灯片和五分钟视频稿，再录制视频、发布经过团队审阅的版本并提交表单。
