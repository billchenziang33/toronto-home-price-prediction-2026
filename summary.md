# 最新状态更新（2026-10-29，Toronto）

- 正式工厂当前选择 Off-market `R52`、List-aware `R97`。选择依据是官方标准六个月（2025-12至2026-05）回顾性walk-forward中最低的MdAPE；该窗口曾用于开发，不是独立holdout。R52：MdAPE 7.06819266%、MAPE 9.52229909%；R97：MdAPE 3.93961679%、MAPE 5.02935657%。所有候选细分结果、置信区间及误差分析见`reports/validation_report.md`和`outputs/error_analysis.csv`。
- 最终预测run `submission_20261029T212456Z_056c5580`已完成8/8 fits，cutoff `2026-06-01`，训练150,000行。`submissions/submission_off_market.csv`（3,974行，R52）规范化SHA-256 `bcb7ff2ac262161a7f3726621259c1ea1abfa6eaaad09191137cd1cb1f072c3f`；`submission_list_aware.csv`（4,064行，R97）规范化SHA-256 `c10e5b1229106f2aa3a368b18a9da0f3dde09cf4531cd37f4f59ebf840a390fd`。官方`run.py check`通过；manifest的数据、源文件及CSV哈希均核验一致，隐藏/reserved标签访问标记为false。
- 2026-10-29 2025-12单月预检完成：五条路线各1,238行，共6,190行，价格有限且为正，ID按路线唯一；manifest确认训练数据SHA为`18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`，未读取reserve/hidden标签。仅该月的MdAPE：Off-market R22 7.2691%、R52 7.4516%、R151 7.4076%；List-aware R35 3.7499%、R97 3.6550%。此单月是管线预检，不代表六个月结果或独立测试成绩。输出在`outputs/submission/preflight/202512/`。
- R48/R50按房型/价格组选择的是逐房权重，旧`r40.blend()`只接收标量。预检中发现该接口不匹配后，改用校验过的逐房blend函数并增加回归测试。另将每次运行ID写入fit账本和run manifest，避免重复fit记录无法区分。10项聚焦测试通过，改动文件已`py_compile`通过；五份安全OOF资产哈希验证通过，R52及R151的15,642条预测精确重建。
- 六个月正式回顾验证run为`validation_20261029T164023Z_cff43dc1`，共50,345行（五路线各10,069行），0重复ID、0无效价格；manifest记录数据SHA `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`，reserved/hidden标签未读取。最终交付说明见`docs/SUBMISSION_TECHNICAL_HANDOFF_2026-10-29.md`；按MdAPE刷新后的候选说明见`docs/Step 1.md`。本地路径已从验证与提交manifest改为仓库相对路径，哈希未变。
- 官方`rubric.md`规定一份最多4页PDF，通过表单提交并链接不超过5分钟的YouTube视频（unlisted可、private不可）和公开或与组织方共享的GitHub仓库；仓库需含源代码及两份预测CSV。独立slides不是必交项，可在视频中用。Codex已完成技术文件；PDF、录制与上传视频、最终表单提交由团队成员处理。

# 历史状态（截至2026-10-28）

- 用户最新将下一阶段范围限定为Codex可独立完成的技术交付：模型接口整合、可复现时间验证、预测CSV及官方检查、误差分析、技术Markdown/manifest、README/summary和bill分支代码交接。PDF排版、slides、讲稿、视频、口头展示、表单及队友沟通均为人工工作，不作为Codex验收条件。目标文档与桌面副本需保持SHA-256一致；计划稿不代表正式管线整合或新验证已经启动。

- 新目标为11月7日20:00 EST前交付可审计的技术包；比赛截止为11月8日23:59 EST。正式工厂仍为旧HGBR，R151/R97尚未接入；requirements显式依赖、官方六个月验证、预测CSV、误差分析及干净环境复跑仍待完成。R151冻结，R97不确定性和<4%研究愿望需如实说明。无新增fit/时长上限；截止日是日程而非训练预算。

- 2026-10-28技术盘点记于 `docs/SUBMISSION_TECHNICAL_AUDIT_2026-10-28.md`：官方 `make_model` 仍为 HGBR；隐藏输入 schema 行数分别为 off-market 3,974、list-aware 4,064，盘点时只核对 schema。R32 campaign 为 `screen_failed`、R52 为 `development_failed`；R151/R97分数和晋级限制须同时披露。当前分支为 `bill`、HEAD `81766ae`，保留大量既有未提交文件。候选分数：R151 MdAPE 6.9988809893%、MAPE 9.5180175461%；R97 MdAPE 4.5440739072%、MAPE 5.7942660284%。

- TDD修正 `models.make_model`：未知track原先会静默落入off-market HGBR；新增 `ValueError` 并在 `tests/test_submission_models.py` 覆盖。红灯复现：`DID NOT RAISE ValueError`；实现后绿灯：`1 passed`。合法track目前仍返回HGBR；R151/R97路线尚未接入，未做训练或最终提交预测。

- 2026-10-28新增 `tools/build_submission_assets.py` 和 `model_assets/` 安全OOF包：R151、R97各15,642行；补充R47严格过去权重输入19,604行、R48/R50对齐输出15,642行及R151候选逐房输出。R47历史结果为 `development_failed`，以保存的源码快照/结果哈希及精确安全月训练标签独立核验；未改HEAD。manifest记录训练快照、运行来源和SHA-256；reserved月份与隐藏标签未进入构建。`verify_assets()`在不打开训练数据时验证全部5个资产通过；聚焦测试刚更新，待复跑。此包仍是研究/校准输入，未生成正式提交CSV；官方`run.py`仍用HGBR，R52为 `development_failed`、R97未通过所有晋级门槛。详见 `docs/SUBMISSION_TECHNICAL_AUDIT_2026-10-28.md`、`model_assets/README.md`、`model_assets/manifest.json`。

- 已将 `docs/Step 1.md` 改为按官方主指标 MdAPE 选基准：Off-market R151（MdAPE 6.9989%、MAPE 9.5180%）、List-aware R97（MdAPE 4.5441%、MAPE 5.7943%）。同步更新模型方法、比较基准、月度/价格/房型误差和局限说明；明确 R97 的配对区间跨 0、MAPE 微升，不能称稳健晋级。工作稿不等于官方隐藏测试或最终提交，桌面 SSU Data Competition/Step 1.md 已同步且哈希一致；模板 `report/report.tex` 仍待同步。

- 比赛要求复核：本地 `handout.md`、`rubric.md`、`report/report.tex` 均保留。官方准确度占 30/100（Off-market 20、List-aware 10），按隐藏集 MdAPE 排名，MAPE 用于同分比较；获得赛道准确度分的门槛是在同一隐藏房屋上优于官方简单基线，没有固定 7%/4% 合格线（这是本项目内部目标）。当前 R151/R97 的 6.9989%/4.5441% 为反复使用的八个月开发分数，不能直接当作最终隐藏成绩或与官方 2025-12 至 2026-05 六个月基线作等条件胜负结论。最终交付需冻结模型、对齐官方时间验证及两份预测 CSV，并说明已记录的 reserve 访问事件；不能把已接触月份称作未触碰 holdout。当前判断：已有值得呈现的研究成果，但获奖排名、最终泛化与完整提交可复现性仍未由这些分数确认。

- 用户最新调整：Off-market R151 已达标并冻结，接下来只推进 List-aware，目标仍为八个月开发 pooled MdAPE <4.00%。本次完成 List-aware 实验清单 `docs/LISTAWARE_EXPERIMENT_INVENTORY_2026-10-28.md`：登记 136 张相关卡、130 张有数值评分、14 个八个月开发候选；R97 当前最佳 4.5440739% MdAPE / 5.7942660% MAPE，目标尚差 0.5441 个百分点。完整统计按候选卡去重，筛选分数和开发分数分开标注。

- R195 List-aware 残差训练目标限幅放宽到 `±0.20` 已完成：R35 固定双月 pooled MdAPE `4.7274305%→4.8762532%`（相对恶化 `3.1481%`），MAPE `5.9491287%→6.2011242%`（恶化 `4.2358%`），两月均退步；相对 R97 同支持 MdAPE `4.8504910%→4.8762532%` 也恶化 `0.5311%`。配对 MdAPE 差值 95% CI `[+0.06322,+0.25034]` pp。审计 `17/17` 通过，筛选/审计 `4/4` fits 完成，未跑八月开发。R195 未读取 reserve/hidden；campaign reserve 验证资格仍因更早记录的访问事件为无效。详见 `docs/R195_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，screen `r195s_20261028T224412Z`。
- R195 运行物理 RAM 可见总量 `15.8285 GiB`；screen/audit 最低可用 `2.0873/2.5511 GiB`，最高负载 `86%/83%`，进程工作集峰值 `0.2429/0.2394 GiB`。登记、筛选和审计运行日志均正常结束。
- R194 (`±0.10`) 与 R195 (`±0.20`) 两种残差训练目标限幅都未胜 R35/R97，停止在 R161 路线上继续扫该参数。下一候选方向是按安全过去 OOF 记录、按房屋类型分别选择 R35/R97 修正混合权重；需设定每组最低历史量与全局回退，并在固定双月筛选、独立审计通过后才跑八月开发。
- R194 List-aware 残差训练目标限幅 `±0.10` 已完成：R35 固定双月 pooled MdAPE `4.7274305%→4.8776334%`（相对恶化 `3.1773%`），MAPE `5.9491287%→6.2070300%`（恶化 `4.3351%`），2023-03 与 2024-06 两月均退步；相对 R97 同支持 MdAPE `4.8504910%→4.8776334%` 也恶化 `0.5596%`。R35 配对 MdAPE 差值 95% CI `[+0.07389,+0.26879]` pp。独立审计 `17/17` 通过，筛选/审计 `4/4` fit 完成，未跑八月开发。R194 运行标记未读取 reserve/hidden；但 campaign 的 reserve 验证资格因先前记录的访问事件仍为无效。详见 `docs/R194_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，screen `r194s_20261028T223307Z`。
- R194 运行期物理 RAM 可见总量 `15.8285 GiB`，screen/audit 最低可用 `1.9689 GiB`、最高负载 `87%`，候选/审计工作集峰值约 `0.2300/0.2420 GiB`；运行日志均以 `runtime_complete` 结束。
- 下一项 R195 只把 R161 训练残差目标限幅从 `±0.15` 放宽到 `±0.20`，预测端限幅仍为 `±0.10`，其他配置和门槛不变。安全严格过去 OOF 中，残差绝对值超过 `0.15` 的比例为 `231/3,962`（2023-03）与 `780/12,313`（2024-06）；超过 `0.20` 的为 `62` 与 `260`。这是一个尾部标签信息假设，须经固定双月筛选与独立审计；失败不跑开发，继续保留 R97。
- R193 List-aware 扩充严格过去 OOF 历史已完成并通过独立审计，但候选未通过双月 screen：R35 pooled MdAPE `4.7274305%→4.8701920%`（相对恶化 `3.0199%`），MAPE `5.9491287%→6.1871723%`（相对恶化 `4.0013%`），2023-03 和 2024-06 两月都退化；配对 MdAPE 差值 95% CI 为 `[+0.06299,+0.24220]` pp。21/21 审计检查通过，包含 3 个早期父模型 OOF 折独立重建；R193 共 `22/22` fits 完成，没有八个月开发运行，也未读取 reserve/hidden 标签。当前最优仍是 Off-market R151 `6.9988809893%`、List-aware R97 `4.5440739072%`。详情见 `docs/R193_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，screen `r193s_20261028T220330Z`。
- R193 审计运行时物理内存可见总量 `15.8285 GiB`，可用最低 `0.3053 GiB`、负载最高 `98%`，审计进程工作集峰值 `0.5516 GiB`；系统 commit 峰值 `27.2895/39.4881 GiB`（包含页面文件）。运行结束时可用 `1.8899 GiB`、负载 `88%`，没有观测到物理使用超过容量。记录见上述 post-run review 和 `outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly/runtime_20261028T220443Z_22652.jsonl`。
- 下一项只改 R161 的训练目标残差限幅：把 `clip(log(sold_price/R35_prediction), -0.15, 0.15)` 改为 `[-0.10,+0.10]`，其他特征、CatBoost、历史、预测修正和门槛保持不变。安全历史 OOF 中，绝对残差超过 0.10 的比例在两个 screen 训练历史分别为 `824/3,962 (20.8%)` 和 `2,342/12,313 (19.0%)`；这是值得检验的机制假设，不预设会提升。先登记并预检，再跑固定双月 screen 与独立审计；失败则不跑八个月开发。
- R188 List-aware 全历史 CatBoost L2 消融已完成：仅将 `l2_leaf_reg` 从 20 降至 10，保持 R161 的 3,500 行严格过去 OOF 门槛、seed `20261025`、R97 特征/目标/修正不变。筛选 pooled MdAPE 对 R35 从 `4.7274305%` 升至 `4.8870764%`（相对恶化 `3.3770%`），MAPE 从 `5.9491287%` 升至 `6.2049273%`（恶化 `4.2998%`），两月均变差；配对 MdAPE 差值 95% CI `[+0.0770,+0.2550]` 个百分点。对 R97 pooled MdAPE `4.8504910%→4.8870764%`（相对恶化 `0.7543%`），仅 2024-06 小幅改善。2023-03 有 3,962 条历史，低于 R97 原始 4,000 行门槛，因此对 R97 的 pooled 比较存在一折拟合支持差异；R35 是主要对照。独立审计 17/17 项通过；2 个筛选拟合和 2 个审计重训全部完成。未跑八月开发期，未读取 reserve/hidden 标签。当前最优保持 Off-market R151 八月 pooled MdAPE `6.9988809893%`、List-aware R97 `4.5440739072%`。详情：`docs/R188_SCREEN_POSTRUN_REVIEW_2026-10-28.md`；运行 `r188s_20261028T203354Z`，审计 `outputs/bill_target74/bill_target74_unbounded_20261024/batches/r188s_20261028T203354Z/independent_audit.json`。
- 用保存的 R161 同支持预测做了无拟合的直接对照，确认两个筛选折样本和查询行完全一致，除 L2 外训练配置一致：R161 L2=20 pooled MdAPE `4.8939212%`，R188 L2=10 `4.8870764%`，微幅相对改善 `0.1399%`；仅 1/2 月改善，配对 MdAPE 差值 95% CI `[-0.0389,+0.0244]` 个百分点跨 0，不构成可靠优势。此诊断已补入 R188 post-run review 和协议卡；无额外 fit。
- R188 按用户要求在两项 RAM 等待门槛均为 `null` 时运行；运行器 BelowNormal、2 个逻辑核、每 15 秒采样。Windows 可见物理 RAM `15.8285 GiB`；screen/audit 合计最低可用 `1.8294 GiB`、最高负载 `88%`，候选/审计进程工作集峰值分别 `0.2538/0.2422 GiB`。系统提交量峰值 `24.8794/39.4881 GiB`（含页面文件，不等于实物 RAM）。没有观测到物理使用超过 `15.8285 GiB`；可用内存较低主要是系统总体占用，不应归因于 R188 单进程。运行日志见 post-run review。
- R189 List-aware CatBoost L2=40 已完成：R35 pooled MdAPE `4.7274305%→4.8979745%`（相对恶化 `3.6075%`），MAPE `5.9491287%→6.2111409%`（恶化 `4.4042%`），两月均变差；配对差值 95% CI `[+0.0767,+0.2641]` 个百分点。对 R97 pooled MdAPE `4.8504910%→4.8979745%`，相对恶化 `0.9789%`。对同支持 R161 L2=20，R189 pooled MdAPE `4.8939212%→4.8979745%`（恶化 `0.0828%`），两月均变差、配对区间跨 0。独立审计 17/17 项通过；2 个候选及 2 个审计 fit 全部完成；不跑八月开发，R97 仍是最优。详情 `docs/R189_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，运行 `r189s_20261028T204108Z`。
- R189 无 RAM 等待门槛运行，运行器仍为 BelowNormal、2 个逻辑核、15 秒采样。Windows 物理 RAM `15.8285 GiB`；screen/audit 最低可用 `0.9594 GiB`、最高负载 `93%`，R189 进程工作集峰值分别 `0.2378/0.2420 GiB`。系统提交量最高 `25.1281/39.4881 GiB`，包含页面文件。未观察到物理使用超过可见容量；低可用内存是系统总体压力，R189 自身工作集仍约 `0.24 GiB`。运行日志和详细内存口径见 post-run review。
- R190 List-aware CatBoost depth=3 已完成：R35 pooled MdAPE `4.7274305%→4.8868125%`（相对恶化 `3.3714%`），MAPE `5.9491287%→6.2000773%`（恶化 `4.2182%`），两月均变差，95% CI `[+0.0628,+0.2525]` 个百分点。对 R97 pooled MdAPE `4.8504910%→4.8868125%`，相对恶化 `0.7488%`。对同支持 R161 depth=4，R190 depth=3 pooled MdAPE `4.8939212%→4.8868125%`（微幅改善 `0.1452%`），两月均略改善但配对 95% CI `[-0.0489,+0.0247]` 个百分点跨 0。审计 17/17 项通过，4/4 拟合完成；屏幕门槛失败，不跑八月开发。详情 `docs/R190_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，运行 `r190s_20261028T204812Z`。
- R190 运行期 RAM 门槛均为 `null`。screen 可用物理 RAM 最低 `0.3250 GiB`、最高负载 `97%`、进程峰值工作集 `0.2533 GiB`；audit 最低可用 `1.2828 GiB`、最高负载 `91%`、进程峰值工作集 `0.2393 GiB`。可见物理 RAM 为 `15.8285 GiB`；系统 commit 峰值 `26.0871/39.4881 GiB`（含页面文件）。没有观测到物理使用超过总量。任务结束后 probe 为可用 `2.8632 GiB`、负载 `81%`，说明内存压力在运行器退出后回落。
- R191 List-aware CatBoost depth=2 已完成：R35 pooled MdAPE `4.7274305%→4.8959970%`（相对恶化 `3.5657%`），MAPE `5.9491287%→6.1883266%`（恶化 `4.0207%`），两月均变差，95% CI `[+0.0606,+0.2475]` 个百分点。对 R97 pooled MdAPE `4.8504910%→4.8959970%`（恶化 `0.9382%`）。同支持 R161 depth=4 比较中 R191 MdAPE 略差 `0.0424%`，虽然逐月中位误差小幅改善但 bootstrap 区间跨 0。审计 17/17 项通过，筛选与审计 4/4 fit 完成；无 reserve/hidden 标签访问；screen 失败，不跑八月开发，停止 depth sweep。详情 `docs/R191_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，运行 `r191s_20261028T205812Z`。
- R191 RAM 门槛均为 `null`；物理 RAM 可见总量 `15.8285 GiB`，screen/audit 最低可用分别 `2.1599/2.3858 GiB`，最高负载 `86%/84%`，进程工作集峰值 `0.2416/0.2270 GiB`。系统 commit 峰值 `24.8112/39.4881 GiB`（含页面文件）。未观察到物理使用超过可见容量。
- R192 List-aware 周期月份特征筛选已完成，未晋级。只向 R97/R161 残差特征加入 `sin/cos(2π(m-1)/12)`，其余 CatBoost、目标、seed、校正和历史窗不变。2023-03、2024-06 共 4,662 行上，R35→R192 pooled MdAPE `4.7274305%→4.7012933%`（改善 `0.5529%`），但 MAPE 恶化 `2.0430%`、仅 1/2 月改善，配对差值 95% CI `[-0.1282,+0.0820]` pp 跨 0；2023-03 明显退步、2024-06 改善。对 R97 诊断值 MdAPE 改善 `3.0759%`，但也只改善 1/2 月。独立审计 20/20 通过，候选及审计 `4/4` fits 完成；screen 失败，不跑八月开发，不读取 `2025-05`、`2025-08` 或隐藏标签。保留 R97 当前最佳 `4.5440739%`。详情见 `docs/R192_SCREEN_POSTRUN_REVIEW_2026-10-28.md`，运行 `r192s_20261028T212837Z`。
- R192 按用户要求在导入前和 fit 前 RAM 等待门槛均为 `null` 时实跑。Windows 可见物理 RAM 为 `15.8285 GiB`（名义 16 GB），两次运行合计最低可用物理内存 `1.4483 GiB`、最高 load `90%`；候选/审计运行器工作集峰值 `0.2405 GiB`，系统 commit 峰值 `21.6334/39.4881 GiB`（含页面文件）。运行正常完成，未观察到物理使用超过可见容量。两核、BelowNormal 和单 campaign 进程策略保留。详见 R192 post-run review 与两条运行器日志。
- R193 已按之前的 past-only OOF 研究顺序完成；新增的早期历史并未改善当前 R161 残差学习器。后续沿单机制实验推进，先验证训练残差限幅，再依据结果选择新的方法，不能在失败 screen 上调参。
- R187 List-aware 单位面积要价特征卡已完成两折筛选及独立审计，未晋级。R35 pooled MdAPE `4.7274305%→4.8579458%`（相对恶化 `2.7608%`）、MAPE 相对恶化 `4.0297%`、筛选月 `0/2` 改善；对 R35 的配对 MdAPE 差值 95% CI 为 `[+0.0470,+0.2372]` 个百分点，完全偏向误差升高。对 R97 pooled MdAPE `4.8504910%→4.8579458%`（相对恶化 `0.1537%`），只有 2024-06 小幅改善、2023-03 恶化。R187 以 R161/R97 CatBoost MAE 配置、seed `20261025` 和 `3500` 行历史门槛加入 `log(list_price/area_midpoint)`；2023-03 仅有 `3962` 条历史，低于 R97 原始 `4000` 门槛，故该月对 R97 比较受拟合门槛差异影响；2024-06 两种门槛均满足。独立审计 18 项全部通过，候选/审计 `4/4` fit 完成，无 reserve/hidden 标签访问。基于同一筛选预测的无拟合 shrinkage 网格（0–0.50、步长0.025）没有正值能同时改善两月；最低 pooled MdAPE 为 alpha `0.075` 的 `4.72501%`，但两月仍均略差，不继续该特征加收缩路线。详情见 `docs/R187_SCREEN_POSTRUN_REVIEW_2026-10-28.md`。
- 无 RAM 等待门槛下完成 R187 筛选和审计：Windows 可见物理 RAM `15.8285 GiB`；两次运行合计观察最低可用 RAM `2.3664 GiB`、最高负载 `85%`，运行器最大工作集 `0.2451 GiB`。系统提交量最高 `23.8507/39.4881 GiB`，包含页面文件，不能视作物理 RAM。两项阈值仍为 `null`，运行器实际以 BelowNormal、2 logical CPUs 启动且没有等待内存门槛；未观察到超过 16 GiB 物理容量。日志见报告链接。
- R187 登记运行 `r187s_20261028T202240Z`；累计 campaign ledger 为 `1053` 个 `fit_start`、`1042` 个 `fit_complete` 事件（其中 `1038` 完成、`4` 失败）。历史 fit 事件包含候选、审计和重试，不代表独立模型数量。R187 筛选及审计 JSON 位于 `outputs/bill_target74/bill_target74_unbounded_20261024/batches/r187s_20261028T202240Z/`。
- R184–R186 均完成筛选并通过最终独立审计，但都未晋级；最佳版本仍为 Off-market R151 八个月 pooled MdAPE `6.9988809893%`、List-aware R97 `4.5440739072%`。R184 R97 空间残差校正筛选为 `4.86513%`，比 R35 `4.72743%`差 `2.913%`、MAPE 恶化 `3.188%`；R185 三种子 MAE 平均为 `4.89663%`，比 R35 差 `3.579%`、比 R97 差 `0.951%`，MAPE 比 R35 恶化 `4.249%`；R186 Huber (`delta=0.05`) 为 `4.88866%`，比 R35 差 `3.411%`、比 R97 差 `0.787%`，MAPE 比 R35 恶化 `4.148%`。R186 对 R35 的配对 bootstrap MdAPE 差值 95% 区间为 `[+0.0669,+0.2548]` 个百分点，方向明确为误差升高；两个筛选月均未改善。
- R185/R186 事后方法复核发现一项对比较很重要的口径偏差：两张卡从 R161 继承 `MIN_HISTORY_ROWS=3500`，而 R97 原始卡设为 `4000`。因此 2023-03 的 3,962 条历史行在 R185/R186 中进入了拟合，R97 在该折按原规则回退为父模型；两张卡原先“只改种子/只改损失”的文字不完全准确。2024-06 的历史样本为 12,313 条，超过两个阈值，该折对 R185 的多种子与 R186 的 Huber 改动更具可比性。结果仍保留为候选路线失败证据，但不得把完整两月差异单独归因于种子或损失。复核事件已追加到账本与两张卡。
- R186 初次调用独立审计时，审计器导入了旧的 `r186_listaware_r97_three_seed_ensemble` 文件名，启动 fit 前即失败（0 次审计 fit）。未改写 R186 冻结源代码；使用独立恢复审计器 `experiments/bill_target74/r186_independent_audit_recovery.py` 完成两折重训，源/协议/数据哈希、strict-past、行级重建、指标复算、账本配平和标签隔离共 17 项检查全部通过。屏幕结果 `r186s_20261028T200936Z`，恢复审计 `outputs/bill_target74/bill_target74_unbounded_20261024/batches/r186s_20261028T200936Z/independent_audit.json`。
- 无拟合的 R97 八月 OOF 分组诊断显示：R35 pooled MdAPE `4.57852%`、MAPE `5.78687%`；R97 分别为 `4.54407%`、`5.79427%`。R97 在 2023-03、2023-07 两折逐美元等于 R35（前者 3,962 行未达 4,000 行门槛；后者虽达到门槛但输出四舍五入后未改变），在 2023-11、2024-09、2025-01、2025-03 改善，在 2024-03、2024-06 恶化；逐月表现提示时间/市场阶段差异比单一房型分组更明显。以已审计 R168 历史全局权重融合也未改善：八月 pooled MdAPE `4.58157%`，高于 R35 和 R97。完整分组诊断和候选特征假设现已保存至 `docs/R97_RESIDUAL_ERROR_DIAGNOSTIC_2026-10-28.md`，R187 后续筛选结果另见其 post-run review。
- R186 运行期间系统可见物理 RAM `15.8285 GiB`、最高 load `77%`，可用约 `3.60–3.96 GiB`；筛选/恢复审计运行器工作集约 `0.24 GiB`，没有接近或超过 16 GiB。两项 RAM 等待门槛保持 `null`；BelowNormal、双逻辑核策略照常生效。日志：`outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly/runtime_20261028T200927Z_5960.jsonl`、`runtime_20261028T201103Z_34636.jsonl`。
- R186 方案、计划和实现：`docs/superpowers/specs/2026-10-28-r186-listaware-r97-huber-residual-design.md`、`docs/superpowers/plans/2026-10-28-r186-listaware-r97-huber-residual.md`、`experiments/bill_target74/r186_listaware_r97_huber_residual.py`、`r186_independent_audit.py` 与恢复审计器。预检确认 CatBoost 接受 `Huber:delta=0.05`，语法及安全 fold 预检通过。R186 正式结果已证明该 Huber 候选不晋级；修正方法口径偏差后，下一张卡须以已试过的 CatBoost MAE/R161 和 `3500` 行门槛为可比基线，不重复归因阈值变化。
- Campaign 累计到账本 `1,049` 个 `fit_start`、`1,038` 个 `fit_complete` 事件；R184 共 7 个 fit 事件（含恢复审计尝试），R185 `12/12` 完成，R186 `4/4` 完成。事件数包含候选和审计重训，不是独立模型数量。R184–R186 均未读取 reserve/hidden 标签。

- R180–R183 四个 list-aware 筛选均已完成独立审计，审计检查全通过，但四者都未通过晋级门槛；当前最佳模型不变：Off-market R151 八个月 pooled MdAPE `6.9988809893%`，List-aware R97 `4.5440739072%`。R180 Detached garage-count：筛选 `4.72743%→4.70127%`（+0.553%相对改善），但 2023-03 从 `4.59667%`退到`4.62926%`，因此未过“两月均改善”门槛。R181 365天半衰期样本权重：`4.72743%→4.89986%`，MAPE 相对恶化 `4.53%`；R182 180天半衰期：`4.72743%→4.89247%`，MAPE 相对恶化 `4.51%`，两者配对 bootstrap 的 MdAPE 差值 95% 区间均完全大于 0。R183 直接拟合 `log(sale/list)`：筛选 `4.72743%→5.19116%`、MAPE 相对恶化 `10.97%`，明确不采用。四卡共 8 次筛选 fit + 8 次独立重建审计 fit，16/16 完成；模型拟合累计用时分别为 R180 `330.3s`、R181 `32.5s`、R182 `33.3s`、R183 `36.1s`；账本未记录 reserve/hidden 标签访问。R182 审计 JSON 中的 `change_note` 是从 R181 审计代码沿用的历史注释，指 R181 一次发生在 fit 前的导入错误，不代表 R182 审计失败；R182 本次所有审计检查均通过。
- 按用户要求，桌面运行器的两项 RAM 等待门槛持续为 `null`。R180–R183 运行期间 Windows 可见物理 RAM 总量 `15.8285 GiB`，最低可用 `1.8356 GiB`、最高负载 `88%`；单个训练/审计运行器最大工作集约 `0.5149 GiB`（R180），R183 本轮最大约 `0.2591 GiB`。没有观测到物理内存超过 16 GiB；系统提交量包含页面文件，不能当成物理 RAM。最新快照可用 `2.5026 GiB`，负载 `84%`。运行器仍是 BelowNormal、2 个逻辑核；各日志位于 `outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly/`。
- R183 方法与步骤已记录在 `docs/superpowers/specs/2026-10-28-r183-listaware-direct-log-ratio-design.md`、`docs/superpowers/plans/2026-10-28-r183-listaware-direct-log-ratio.md`；代码为 `experiments/bill_target74/r183_listaware_direct_log_ratio_catboost.py` 与 `r183_independent_audit.py`，运行 `r183s_20261028T194028Z`。R183只改变为直接预测 `log(sold_price/list_price)`，保留 R97 特征和 CatBoost 参数；两个筛选月都比 R35/R97 差。当前建议暂停再换目标或权重，下一步先只读检查 R97 OOF 残差是否存在按房屋类型/空间位置稳定的互补结构，再决定是否登记 R97 加局部空间残差校准的新卡；继续使用严格过去 OOF、两月筛选、独立审计与 reserve/hidden 隔离。
- Campaign 账本现为 `1,026` fit starts、`1,015` fit_complete 事件；历史上有未闭合/失败记录，不能把事件数当作独立模型数。R180–R183 各卡分别有 4 次 fit start、4 次成功完成（候选 2 + 审计 2），均为 `screen_failed`、`audit_passed=true`。

- 用户要求移除 R179 的模块导入前和每次 fit 前内存等待门槛，并按无门槛方式实跑。`desktop_friendly_policy.json` 两项阈值现为 `null`，保留 BelowNormal/2 逻辑核/线程限制；运行器采集系统提交量及进程工作集。R179 v3 在 18:44:08 UTC 开始，实际开始 fit，没有被旧门槛阻挡；截至退出 18:52:06 UTC，运行器工作集峰值约 0.63 GiB，私有提交峰值约 0.60 GiB。系统可见物理 RAM 15.8285 GiB，观察到可用最低 1.465 GiB、load 峰值 90%（约使用 14.36 GiB），未超过 16 GiB 物理容量；Windows 系统提交峰值约 28.00/39.49 GiB，包含页面文件，不等于物理 RAM。此次审计完成 5 项重建 fit 后因 List-aware 2023-03 的逐位预测断言失败而退出，最大差仅 `9.313225746154785e-10` 美元；完整审计未通过/未完成，且 R179 原两月筛选仍未晋级。日志：`outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly/runtime_20261028T184408Z_16976.jsonl`；详情：`DESKTOP_FRIENDLY_RUNTIME_DIAGNOSTIC_2026-10-28.md`。命令沙箱无法列举桌面全部进程，整机提交量短时波动不能归因给单一进程。
- 本机内存详细诊断已保存到 `outputs/memory_diagnostic/MEMORY_DIAGNOSTIC_2026-10-28.md`，附宿主机快照及 Docker/WSL 证据。确认物理 RAM 16.00 GiB、Windows 可见总量 15.8285 GiB。14:31 Toronto HIC `hic_frozen_resume.py` PID 25172 工作集约 1.91 GiB；房价 R179 PID 19376 的等待进程在 14:34 达到导入门槛并启动运行器，14:35 因代码要求的 Git `parent_commit` `0802cc8d...` 与当前 HEAD `81766aec...` 不一致而 `runtime_failed`，没有进入 R179 v2 的 audit fit。此退出与 4.5 GiB 门槛无关。Qdrant 的 guest RSS/缓存与 Windows `vmmemWSL` 嵌套，不可直接相加。
- R179 的 `minimum_available_memory_before_module_gib=4.5` 是固定的系统可用物理 RAM 门槛；普通每次 fit 门槛是 4.0 GiB。旧运行器观察到模块/数据准备前后可用 RAM 约从 4.20 降至 3.82 GiB（下降约 0.38），所以 4.5 预留后约 4.12 GiB，留出约 0.12 GiB 高于 fit 门槛。这是按旧运行观测设的保守缓冲，并非经峰值剖析得出的硬性模型需求；详见 `DESKTOP_FRIENDLY_TRAINING_2026-10-27.md` 与 `experiments/bill_target74/desktop_friendly_policy.json`。
- 2026-10-28 14:19 Toronto 按用户要求执行内存优化复核：Bilibili 视频原先暂停，但 Alt+F4 仅关闭窗口、6 个后台进程仍占约 220 MiB 私有常驻；核实路径后结束其进程树，复核进程数为 0。随后连续三次可用物理 RAM 为 1.873、1.896、1.894 GiB；HIC PID 25172 工作集稳定约 1,686 MiB，R179 PID 19376 约 14.7 MiB。可用内存仍低于 R179 的 4.5 GiB 门槛；因 HIC 在前后测量间启动/恢复，不能把前后全部变化归因于 Bilibili。Edge 的 GitHub 仓库页和 Qdrant 监控页保留。当前对话没有活动子代理；没有批量结束 Codex/Node 进程。WSL 2.7.3 和 Docker/Qdrant 未重启或改配置，以免中断 HIC；WSL 回收设置留待维护窗口。
- 用户询问 16 GiB 的具体占用后，2026-10-28 14:22 Toronto 完整快照 `outputs/memory_diagnostic/memory_snapshot_20261028T182147Z.json` 显示 Windows 可见 15.8285 GiB、可用 1.7979 GiB；各进程私有常驻工作集合计约 7.21 GiB（不是总物理占用）。最大组为 Codex 桌面 `ChatGPT.exe` 1.83 GiB、`codex.exe` 1.10 GiB、Python 0.71 GiB、`vmmemWSL` 0.57 GiB、Edge 0.51 GiB。HIC PID 25172 工作集 1.89 GiB（其中私有常驻 0.70 GiB）；房价 R179 PID 19376 工作集约 14 MiB。Windows 提交量 29.98/39.49 GiB（75%），页面文件当前使用约 1.58 GiB，均不等于 16 GiB 实物 RAM；Qdrant Docker guest 指标与 Windows `vmmemWSL` 嵌套，不相加。明细和口径解释追加至内存诊断报告。
- 14:28 快照显示 CC Switch 已不在运行；Naiwa 桌面宠物私有常驻约 49 MiB，完整工作集约 172 MiB。按用户要求只关闭了 Naiwa 窗口；14:31 检查 `Naiwa.exe` 已退出，Bilibili/CC Switch 也均无进程。14:31 单次可用内存读数为 2.524 GiB（约 13.305 GiB 不在“可用”计数中）；与此前快照的差异受内存波动影响，不能全归因于 Naiwa。Edge、HIC、R179、WSL/Docker 保留。任务管理器约占私有常驻 91 MiB，暂留作当前监控窗口；关闭它不会影响两项任务。
- 交接报告的数据 SHA-256 已修正为 `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`，与 R151、R97 的运行清单及 campaign 协议一致；之前报告中的两个哈希字符串误写，不影响已保存的模型指标。修正范围仅为报告和此记录。
- R179 审计等待期间完成只读指纹预检：`r179a_20261028T045521Z/run_manifest.json` 所列 17 个代码及外部数据文件、`screen_results.json` 和 `paired_predictions.csv` 的 SHA-256 均与清单一致。此检查不代替独立重建审计，也不代表模型拟合已经开始。
- 使用已保存的八个月开发预测完成 R97 list-aware 修正幅度的只读描述性分组：4,476 行没有修正；绝对对数修正 (0, 0.01] 的 6,393 行组内 MdAPE 3.9709%→3.9602%，(0.01, 0.03] 的 4,767 行为 5.1667%→5.2080%。这是事后分析，不能据此直接挑选阈值或声称未来收益；详情、来源哈希和完整组别见 `docs/R97_CORRECTION_DIAGNOSTIC_2026-10-28.md`。无新增 fit。
- 2026-10-28 17:25 UTC 再次轮询确认 session 12387 仍活动；R179 审计仍在 `before_module_import` 等待内存，最新记录可用 0.749 GiB，门槛 4.5 GiB，累计等待约 43 分钟，尚无该次审计 fit start。源代码与结果文件指纹预检已通过。R180 尚未排队，应在审计结束且结果通过核查后启动。
- 目标：持续优化 Off-market 与 List-aware，使八个月开发窗口 pooled MdAPE 分别严格低于 7% 与 4%。Campaign `bill_target74_unbounded_20261024` 按用户已确认的无上限预算运行，并继续保留逐 fit 账本、失败记录、逐月指标、代码/数据哈希及标签隔离。
- 当前最佳八个月开发观察值为 Off-market R151 `6.9988809893%`、List-aware R97 `4.5440739072%`。R151 的 paired-home bootstrap 95% CI 完全低于 0 并通过八个月开发门槛；R97 的 CI 跨 0、MAPE 略升且 MdAPE 未低于 4%，仍是 best observed point estimate 而非确认升级。逐月表、数学定义、演示口径和限制已写入 `docs/BILL_CURRENT_TWO_TRACK_MODELS_2026-10-28.md`；对应完整 JSON 证据位于 `docs/model_evidence/2026-10-28/`。
- Campaign 截至本次记录包含 170 个编号 R 实验卡（最高 R179），149 个卡有 fit_start；账本记录 1,005 fit starts、991 个 `fit_complete/status=complete`、3 个 `fit_complete/status=failed`，另有 2 次 interrupted、1 次 abandoned 和 6 条 reconciliation 事件。Fit 事件不等于独立模型数，包含校准、模型拟合与重试。R179 利率特征两月筛选未晋级；R180 尚未登记。
- 约 16:35 UTC 的任务管理器截图显示整机内存使用 83–84%，Codex 应用组约 2.75–2.97 GB；CPU 读数在截图间显著波动。由于命令环境的 `Get-Process` 只看到沙箱 shell、CIM 查询被拒绝，无法归因其余约 180 个进程的占用。训练运行器本身没有启动任务管理器的代码；Computer Use 只用于辅助查看，随后被用户按 Escape 停止。
- R179 v2 日志最后记录为 16:18:28 UTC `resource_sample`（0.593 GiB 可用、CPU 10.14 秒）；R180 最后记录为 16:18:32 UTC `waiting_for_campaign_lock`（CPU 6.36 秒）。截至本次复核两份日志均没有终态事件且超过 17 分钟未更新；受限环境不能确认相关 PID 仍运行或已退出。不要仅凭锁文件存在判断锁仍被占用，也不要删除锁文件或结束未知进程。
- 16:41 UTC 用相同的 Windows 字节区间锁机制探测，campaign 锁为空闲。16:42 UTC 已用 `.venv-opt/Scripts/python.exe` 启动新的 R179 v2 审计进程 PID 19376；日志 `outputs/bill_target74/bill_target74_unbounded_20261024/desktop_friendly/runtime_20261028T164217Z_19376.jsonl` 首条记录为 2.957 GiB 可用，低于 4.5 GiB 模块导入门槛，正在门槛前等待，尚未导入实验模块或开始 fit。统一执行 session 12387 已轮询确认仍活动；新等待逻辑不持有 campaign 锁。
- 16:44:17 UTC 的最新运行器事件为 3.086 GiB 可用；随后内存快照为 2.760 GiB / 82%，到 16:45:21 UTC session 12387 仍活动，累计 CPU 0.3125 秒，尚无 fit start。
- 16:46:17 UTC 的等待事件为 2.896 GiB 可用；独立快照为 3.071 GiB / 80%。Campaign `fits.jsonl` 中 `independent_audit_postrun_v2` 仍是 0 fit starts / 0 completes；16:46:52 UTC session 12387 仍活动。
- 16:54:19 UTC 的等待事件为 3.569 GiB 可用（总内存 15.829 GiB、77% 使用），仍低于 4.5 GiB 模块导入门槛；session 12387 仍活动，累计 CPU 约 0.36 秒，R179 审计尚无 fit start。该等待进程没有改变 R151/R97 的已完成模型结果。
- 已确认旧运行器先拿全局锁、再等待 4.5 GiB 导入内存，因此 R179 等待时会阻塞 R180。已调整 `experiments/bill_target74/desktop_friendly_runtime.py`：先应用低优先级/线程策略，再等内存后拿锁；拿锁后复查，内存不足就释放并继续等；锁等待日志降为每分钟一次。内存阈值、拟合逻辑、模型参数和预算都不变。AST 语法检查通过；当前只有低优先级等待进程，尚未开始模型 fit。
- 资源恢复后先完成 R179 独立审计；若旧进程已结束，用新的 run id 保留历史日志，再按序运行 R180 的两月 list-aware garage-count 筛选。只有筛选门槛通过才进入八个月开发比较。reserve validation 仍不具备资格，禁止读取预留或隐藏标签。
- 详细诊断见 `DESKTOP_FRIENDLY_RUNTIME_DIAGNOSTIC_2026-10-28.md`；更早的桌面运行方案及完整历史见 `DESKTOP_FRIENDLY_TRAINING_2026-10-27.md`、`BILL_CONTINUOUS_GOAL_LT7_LT4_REFRESH_AFTER_R12_2026-10-24.md` 和活动 campaign 输出目录。

---
# Project summary

## 拟定下一阶段：二十种方法、三次累计升级（2026-10-18；计划，未执行）

用户希望在已完成 v2 优化的基础上继续研究，而不是立即进入提交文稿阶段，并确认以本项目 summary 与实验账本作为 Process Book。已写详细执行稿 `docs/superpowers/plans/2026-10-18-three-checkpoint-model-research.md`，可直接复制的目标在 `NEXT_MODEL_RESEARCH_GOAL.md`。本次只准备目标与计划：没有启动新训练，没有调用 create_goal/update_goal 替换活动目标，也没有改动 v1/v2 模型、CSV 或历史分数。最终提交文稿对齐安排在模型版本确定之后。

### 新目标及判断标准

- 以当前 v2 为 C0：off-market MdAPE `9.1733402669%` / MAPE `11.9803440483%`，list-aware `4.2356822010%` / `5.3575403746%`。官方 starter 和旧 HGBR 只作历史参照，本轮新增收益必须对照 C0。
- 审查二十种不同机制，争取至少三种方法有独立增益，并争取 C0→C1→C2→C3 三次累计开发升级。**独立击败 C0 的方法数量、相对直接父版本的累计升级次数，分别记账。** 换参数、随机种子、取整或修复代码不算新方法/新突破。
- 每次双轨开发升级要求两轨各相对父版本降低至少 1% MdAPE，四开发月至少 3/4 月改善，单月退化≤5%、MAPE退化≤1%，并通过配对区间和复现检查。仅一轨改善单列，不凑双轨次数。
- 最终争取两轨相对 C0 各降低至少 3%，约达到 `≤8.898140% / ≤4.108612%`；最低双轨采用幅度仍为各 2%，约 `≤8.989873% / ≤4.150969%`，并通过逐月、MAPE、配对区间与导出门槛。三次各 1% 累计约 2.97%，不能误写成三个百分点。
- 总上限建议为 **240 分钟或180次真实学习器/校准器 fit**，先到者停止。包含代码、特征、内层 OOF、失败、复现、导出与文档；最多4线程、一次只训练一个模型，最终预留至少60分钟。三次升级是研究目标，不是无限运行条件；0/1/2/3次均可如实结束。

### 方法与推进顺序

二十张固定方法卡详见执行稿：M01近期成交加权；M02滚动24月窗口；M03周期时间特征；M04面积区间与log面积；M05房间/规模交互；M06房型原生类别；M07合成空间网格；M08按时点收缩历史估值；M09历史市场动量；M10过去OOF局部残差；M11Huber损失；M12面积归一化目标；M13直接价格MAPE目标；M14全局OOF校准；M15房型收缩校准；M16样条Ridge；M17ExtraTrees；M18XGBoost；M19CatBoost ratio；M20简单互补融合。它们均是待验证假设，不是已证明有用的方法。

1. **先审计可比性。** 保存v1/v2哈希与未取整分数，断言训练损失实际生效，检查过去OOF权限、融合取整与两层缓存。旧L1参数覆盖错误不得重现。
2. **生成必要的C0折外对照。** 六月/十一月用于筛选，七月/十月补充开发检验；二月/四月只提供过去校准种子。按日历cutoff训练，任何历史成交统计对训练行也必须严格as-of。
3. **二十方法逐卡筛选。** 一张卡先固定一套参数，实际运行和未运行分开记；每轨最多四个候选进入四个月开发比较，选择C1。
4. **逐次检查新增机制。** C2、C3每次只向直接父版本增加一个互补机制，保存父子同房预测和增量；失败即保留父版本，不扩大网格凑次数。
5. **锁定后只复核一次六个月。** 2025-12至2026-05已多次使用，始终称回顾性benchmark；不根据这些分数再换模型、权重或目标。隐藏测试只在合格模型导出时输出CSV。
6. **复现、导出和结束。** 全部试验状态、失败、耗时、预测与配置留档；双轨通过才生成独立model_candidate_v3，否则保留v2。更新summary和桌面归档，不把视频、发布、表单当作持续运行条件。

### 上一轮经验如何进入本轮

LightGBM和off-market有限调参有开发证据支持，作为主路线；但list-aware比例目标单独只比同参数raw log-price改善约0.006个百分点，不能预设为强信号。正确L1未胜出，remarks区间跨零，旧garage/lot与comparables整体增益不稳，均不原样重复。新计划强调相对父版本的消融，而不是用最终组合的总增益给每个组件背书。

已核对一个技术细节：v2六个月list-aware混合已取整HGBR，导出脚本则混合原始HGBR再取整；其实际逐房/指标影响仍待执行阶段审计。下一轮先统一评分与推理规则并量化差异，历史成绩不重写，技术修正不计性能突破。校准/目标编码禁止默认随机KFold；最终预测缓存必须包含融合、校准、取整和fold代码。

资料核对使用LightGBM、CatBoost、XGBoost和scikit-learn官方文档，链接保存在执行稿。Agent Reach Exa与版本检查因DNS失败不可用，已用内置联网工具读取官方资料。所有新方法本项目收益仍未知；本次新增真实数据fit为0。

## Current goal: two-track model and prediction-method optimization (completed 2026-10-18)

Execution plan: `docs/superpowers/plans/2026-10-18-two-track-model-optimization.md`; optimization code: `experiments/model_optimization_v1/`; run outputs: `outputs/model_optimization_v1/`. The original frozen release candidate remains unchanged at `release_candidate/`. This run started at 2026-10-18 20:47:29 UTC and ended after 66 real-data fits, within the 72-fit / 180-minute limits.

### Result

The locked LightGBM candidates passed the predeclared two-track adoption rules on the retrospective December 2025–May 2026 walk-forward benchmark. Each track improved in all six months, both MAPE values fell, and the correctly paired home-bootstrap 95% MdAPE-difference intervals stayed below zero:

| Track | Frozen MdAPE → candidate | Relative reduction | Frozen MAPE → candidate | Paired 95% CI, candidate minus frozen |
|---|---:|---:|---:|---:|
| Off-market | 10.5956% → 9.1733% | 13.42% | 13.4001% → 11.9803% | -1.6102 to -1.2453 pp |
| List-aware | 4.3596% → 4.2357% | 2.84% | 5.5651% → 5.3575% | -0.1937 to -0.0601 pp |

Both pass the planned minimum 2% relative MdAPE improvement, at least four of six months improved, no monthly regression above 10%, no MAPE regression above 1%, and paired-bootstrap upper bound below zero. The 5% aspiration was met off-market but not list-aware. These six months were inspected in earlier work and are a retrospective benchmark, not an untouched independent test. The month-cluster sensitivity uses only six clusters and is correspondingly unstable. Hidden-set performance remains unknown.

### Locked methods and experiment record

- The initial three-family development screen selected LightGBM for both tracks. Across June, September, and November 2025, its pooled MdAPE was 10.0368% off-market and 4.4479% list-aware; the same-fold HGBR scores were 10.8787% and 4.5358%.
- The selected off-market configuration is LightGBM preset 6: 1,000 estimators, learning rate 0.05, 31 leaves, 100 minimum child samples, and L2 regularization 5. It predicts log sold price with weight 1.00.
- The selected list-aware configuration is LightGBM preset 5: 800 estimators, learning rate 0.03, 63 leaves, 100 minimum child samples, and L2 regularization 5. It predicts log(sold price / list price) and uses a development-locked 0.75 blend with the same-fold frozen HGBR model.
- Development pooled MdAPE after target/blend selection was 9.5107% off-market and 4.3890% list-aware. The off-market L1/MAE alternative did not win (9.5321%). An initial LightGBM L1 attempt accidentally inherited the RMSE objective; its three fits remain in `fit_ledger.jsonl` but are explicitly excluded. The corrected `regression_l1` fits are separate, counted experiments.
- Real-data fit counts: pilot 4; family screen 14; tune 20; methods 11 (including three corrected L1 reruns, funded by unused screen capacity); locked benchmark, full-data exports, and repeatability checks 17; total 66/72. Six-month benchmark fits were 12. Two fixed-seed November repeats reproduced identical predictions for both tracks.
- Audit confirmed the training parquet hash `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`. Experiments used the frozen feature source, respected track-specific input rules, and did not use hidden sale labels or external real-estate data.

### Candidate and verification

`model_candidate_v2/` contains the loaded model files, a copy of the frozen feature code, exact environment lock, prediction script, benchmark comparison, SHA-256 manifest, and both candidate CSVs. The portable script produced and passed the official submission checker for 3,974 off-market and 4,064 list-aware homes. Model serialization reproduced the in-memory test predictions; the fixed-seed November refits were bit-for-bit identical. Focused tests passed: 16 tests with `.venv-opt\Scripts\python.exe -m pytest -q tests\test_model_optimization.py --basetemp=.pytest_cache\opt-v1-basetemp`.

Detailed evidence is in `outputs/model_optimization_v1/comparison.md`, `benchmark.json`, `screen_leaderboard.md`, `tune_leaderboard.md`, `methods_comparison.md`, `fit_ledger.jsonl`, `reproducibility.json`, and `model_candidate_v2/SHA256SUMS.json`. The full project suite passed 42 tests; the focused optimization suite accounts for 16 of them. The candidate-v2 metrics and CSVs have not yet been integrated into the existing PDF, slide deck, narration, or public GitHub release; those remain the next phase. Keep `release_candidate/` as the archived baseline until the team reviews the updated submission package.

The Desktop archive mirror is `SSU Data Competition/TorontoHomePriceChallenge`; its final optimization sync is recorded in `MODEL_OPTIMIZATION_SYNC_2026-10-18_COMPLETE.json` with SHA-256 hashes for the copied files.

### Recommended next stage

1. Update the English report, six-slide deck, and narration to describe the two selected LightGBM tracks, target transformation, locked blend, and the exact verified benchmark figures. Do not carry forward claims from the older HGBR report without recalculation.
2. Keep the report within four pages and the video within five minutes. Re-render and verify every slide and PDF page; check that each narrated number matches `model_candidate_v2/benchmark_comparison.md`.
3. Assemble the reviewed report, updated slides/script, model code, lock file, and candidate CSVs into one consistent repository version. Check the two CSVs and all source/model hashes again.
4. Record the narration and add the team's actual YouTube URL to the report. Then publish the reviewed version and have one team member submit the form before November 8. Publishing and form submission were not performed in this stage.
## Frozen v1 release-candidate audit — 2026-10-18

The `release_candidate/` folder is the frozen v1 reference: it contains the last public no-remarks source and both prediction CSVs at commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`. The newly evaluated v2 candidate and CSVs are in `model_candidate_v2/`; the v1 report/slides still need to be aligned to v2 before submission. The unpublished remarks model remains a separate experiment and does not provide either candidate's CSVs.

The work order used for this candidate was: (1) extract model code and both CSVs from the same public Git commit; (2) recalculate the published headline scores from saved six-month predictions; (3) correct the paired MdAPE bootstrap on saved remarks predictions; (4) remove category-gain statements that cannot be reproduced from the archived comparison; (5) align the PDF, six-slide deck, English narration, README, evidence, and CSVs to the frozen version; (6) check the PDF, render every slide, run the submission checker, and record file hashes. No features were added and no model was retrained.

The frozen six-month validation metrics reproduce exactly: off-market MdAPE 10.60% (95% home-bootstrap CI 10.38–10.85), MAPE 13.40% (13.17–13.65); list-aware MdAPE 4.36% (4.26–4.46), MAPE 5.57% (5.47–5.66). Each track has 10,069 homes from December 2025 through May 2026. These intervals treat homes as independent and do not cover shared month shocks.

The remarks point estimate is `median(remarks APE) - median(frozen APE)`. The old helper instead bootstrapped `median(remarks APE - frozen APE)`, a different estimand. The corrected, deterministically ordered paired bootstrap gives March–May changes of -0.101 percentage points (95% CI -0.335 to +0.090) off-market and -0.037 (-0.111 to +0.049) list-aware. Both intervals include zero, so remarks stay archived. The old categorical-gain claim (3.80 / 0.19 points with its intervals) is also omitted: it is not reproduced by the saved `feature_v1` / frozen prediction pair, and its exact baseline specification is not preserved.

The candidate includes `report/SSU_DataDash_submission_candidate.pdf` (2 pages; clickable link to the team's pinned GitHub commit; video explicitly pending), `slides/toronto_fourcast_final.pptx` (6 slides; finalizer receipt and six slide renders), `report/video_script_en.md` (477 spoken words, target 4:40 including pauses), source/model explanation in `README.md`, both frozen CSVs, saved validation predictions, corrected remarks archive, metric audits, a verifier, and a SHA-256 manifest. Final checks passed: 22 tests; both `python run.py check` submission checks; exact source/CSV comparison against the frozen commit; saved-prediction metric and remarks audits; six-slide content audit; PDF audit for page size/count, team names, pinned URL, pending-video label, and excluded unsupported category claims; and all 53 manifest hashes. Full results are in `release_candidate/VERIFICATION_RECORD.md`. No retraining was run. Video recording, publishing the approved candidate, and form submission are the next phase.

The sections below preserve the v1 experiment history. Use the result table at the start of this summary and `model_candidate_v2/` for current model metrics and predictions; the existing v1 PDF, slides, and narration are not yet submission-ready for v2.

The verified 54-file candidate was copied to `SSU Data Competition/TorontoHomePriceChallenge/release_candidate/`. All 53 entries in its SHA-256 manifest passed on the Desktop copy. At the candidate handover, the Desktop-wide manifest verified 262 payload files. The later optimization-plan documentation adds two files (264 payload files total); the frozen candidate is unchanged and the archive manifest is refreshed after document synchronization.

## Current status (2026-10-18)

The official starter repository is checked out in this folder at upstream commit `6c5831e`. The local experimental worktree also contains unpublished remarks changes. The candidate separately extracts the release source from public commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`; it uses separate off-market and list-aware feature sets, log-price histogram gradient boosting, and native categorical splits for `house_style`, `municipality`, and `community`.

## Frozen v1 validation result

`release_candidate/` uses six expanding walk-forward months, December 2025 through May 2026, with 10,069 validation sales per track. Its frozen v1 model scores:

| Track | MdAPE (95% home-bootstrap CI) | MAPE (95% home-bootstrap CI) | Median signed error |
|---|---:|---:|---:|
| Off-market | 10.60% (10.38–10.85) | 13.40% (13.17–13.65) | -0.13% |
| List-aware | 4.36% (4.26–4.46) | 5.57% (5.47–5.66) | +0.65% |

The older claim that the three categories reduced MdAPE by 3.80 points off-market and 0.19 points list-aware is not used in the candidate: its reported estimates and intervals do not reproduce from the archived `feature_v1` / frozen prediction pair, and the exact comparison baseline is not preserved. The candidate records the paired-file audit at `release_candidate/evidence/category_ablation_audit.json`. Garage/lot-area and community-comparable experiments had no stable gains and were removed. The first one-hot encoding attempt was too slow; native categorical support produced the successful run.

The official starter baselines were 20.8% MdAPE off-market and 7.3% list-aware. The current results are validation only; hidden-set performance is unknown. The home-level bootstrap intervals do not model shared month-to-month market shocks.

## Verified artifacts and commands

- The frozen candidate source is copied from Git commit `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8`; `release_candidate/tools/verify_candidate.py --source-repo .` confirms code and CSV bytes match those Git objects.
- Frozen validation inputs and derived metrics are in `release_candidate/evidence/validation_predictions_frozen.csv` and `validation_metrics.json`; both CSVs pass `python run.py check` in the candidate directory.
- The local experiment worktree has newer, unpushed remarks changes. Do not pair its current submission CSVs with the frozen report or slides. The release candidate uses only the committed public CSV blobs.
- Verification: `OMP_NUM_THREADS=1 python -m pytest -q --basetemp=.pytest_cache/basetemp` passed 22 tests. The final six-month `validate`, `report`, `predict`, and `check` commands completed. `OMP_NUM_THREADS=1` was used for local reproducibility and runtime control; it does not change the validation splits or metrics.
- Clean-clone reproduction: cloned public `main` at `7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8` into `tmp/repro-check-2026-10-18`, installed pinned `requirements.txt` into an isolated Python 3.13 environment, and ran `OMP_NUM_THREADS=1 .venv-repro/Scripts/python.exe -m pytest -q --basetemp .pytest-tmp-clean`; all 22 tests passed. Pytest's default system-temp location is inaccessible in this managed environment, so `--basetemp` must point inside the clone.
- Data cache: `data/.hf_cache`; set `HF_HOME` to that folder in PowerShell when running locally in this environment.

## Existing v1 report, video, and slides (awaiting v2 alignment)

- The v1 PDF is `release_candidate/report/SSU_DataDash_submission_candidate.pdf`. Poppler confirms 2 Letter pages; PDF text confirms all four full names, the clickable pinned GitHub link, the pending-video marker, and no member email. It reports v1 metrics, so it needs updating before it accompanies the v2 CSVs.
- The v1 deck is `release_candidate/slides/toronto_fourcast_final.pptx`: 6 slides, 16:9 layout, Arial typography, an editable results table, and an embedded static chart generated from the frozen v1 validation CSV. All 6 final slides were rendered and reviewed; their results need to be synchronized with v2.
- The v1 English narration is `release_candidate/report/video_script_en.md`. Its spoken text is 477 words and the planned slide timings total 4:40. Actual recording and duration are still pending.
- The older `output/pdf/` and `output/slides/toronto_fourcast_challenge_slides_v4.pptx` remain historical drafts. Use the `release_candidate/` files for the current claims.

## Submission requirements and limits

The official report must be at most four pages, link to a YouTube video no longer than five minutes and a GitHub repository containing the code and both prediction CSVs, and list every team member's full name. The form requires one team member to submit. The deadline is November 8 at 11:59 pm EST; the in-person final is November 20.

No outside data is used in the model or report. Contest rules restrict outside data to periods through May 2026, ban real listings/sales, and require every number in the video to come from our validation. TRREB's September 2026 market report was reviewed only as background and falls after the allowed data cutoff, so do not use it in the submission or video. Challenge data are synthetic and its communities/coordinates do not match actual Toronto maps.

## Research and presentation guidance

- A Toronto real-estate study describes spatial neighborhood relationships and temporal price changes as useful signals, supporting our test of location categories and time features: https://arxiv.org/abs/2008.05880
- A public Zillow Prize solution extracts region aggregates and interaction features, compares LightGBM/CatBoost, and reports offline cross-validation to reduce leaderboard overfit. It supports the experiment design, but its real Zillow data and error metric are not comparable to this synthetic task: https://github.com/junjiedong/Zillow-Kaggle
- scikit-learn documents native categorical support in `HistGradientBoostingRegressor`; it avoids a large one-hot matrix and is the implementation used here: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- TRREB's September 2026 report says the GTA average sale price was $1,006,409, down 5.1% year over year. This is post-cutoff contextual research only, not an admissible modeling or video input: https://public.trreb.ca/market-data/market-watch/
- The local `Presentations` skill is available for the optional five-minute slide deck/video. Short-talk guidance recommends one main point per slide and a single clear story: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005373

The logged-in Discord desktop window was inspected read-only on 2026-10-18. `#announcements` confirms the Friday, November 20, 2:30–5:00 pm finale; the top five teams present, and every team that submits a video is invited to attend. `#links-and-documents` contains the case handout/rubric and the spatial-modelling workshop slides. Its follow-up announces Workshop 2 for November 2, 1:00–2:00 pm at 700 University Ave, room 9199. `#rules` prohibits academic-integrity violations, spam, off-topic posts, and sharing other people's personal information without permission. Organizers replied that remarks reflect market dynamics and latent-variable distributions; that does not establish an incremental prediction gain over the structured inputs. Rates are not directly modeled, and the generated-market mechanism remains uncertain. No Discord messages were sent.

## Submission steps after model-version review

The model-optimization stage completed and passed both tracks' adoption rules. Continue by reviewing `model_candidate_v2/` and updating every submission artifact to use that same version consistently.

1. Have the team review the v2 model files, configuration, benchmark evidence, and exact CSVs.
2. Record the narration, confirm the video stays below five minutes, insert the YouTube URL in the PDF, rebuild it, and test the link without a login.
3. Publish the reviewed candidate files to the existing public GitHub repository and confirm the uploaded PDF, video, code, and CSVs align.
4. Have one team member submit the official form before November 8 and save the confirmation. Video recording, publication, and form submission are outside this preparation stage.

## Three-checkpoint, 20-method optimization result — 2026-10-19

The user-authorized optimization goal completed in 159.38 minutes with 140 counted real-data fit attempts, below the 240-minute and 180-fit caps. The frozen dataset contains 150,000 rows and its SHA-256 is `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`. The fit ledger records 18 A0, 74 A1, 12 A2, 24 A3, and 12 A4 fits. Twenty candidate method cards were assessed; XGBoost was unavailable in the locked environment, and the ExtraTrees/as-of feature trials that exceeded their development timeout were recorded as such.

The only development checkpoint promotion was list-aware M13 (LightGBM direct-dollar MAPE objective): it improved MdAPE by 2.15% against C0 across the four development months, improved three of four months, reduced MAPE by 0.30%, and passed the paired-home bootstrap criterion. Off-market had no passing C1. None of the three M13-plus-feature bundles (M03, M04, M05) passed the incremental C2 gate for either track. The fixed 50/50 M13/CatBoost blend also failed; C3 was therefore not attempted. No two-track cumulative upgrade was established. The frozen benchmark candidates were list-aware M13 and off-market C0.

The final run evaluated those configurations once on the retrospective December 2025–May 2026 benchmark. This period had already been inspected in earlier work, so it is a retrospective comparison rather than a fresh independent holdout. Each track contains 10,069 paired homes.

| Track | Frozen C0 MdAPE / MAPE | Frozen candidate MdAPE / MAPE | Final-gate result |
|---|---:|---:|---|
| Off-market (C0) | 9.1733% / 11.9803% | 9.1733% / 11.9803% | Fail: no improvement |
| List-aware (M13) | 4.2357% / 5.3575% | 4.2496% / 5.4441% | Fail: MdAPE worsened 0.33%; MAPE worsened 1.62% |

List-aware M13 improved in three of six months, short of the four-month requirement. Its worst monthly MdAPE regression was 3.76%, within the 10% limit, but the paired-home 95% interval for the candidate-minus-C0 MdAPE difference was -0.0589 to +0.0863 percentage points and included zero. M13 also exceeded the permitted 1% MAPE regression. Because off-market remained at C0 and list-aware M13 failed, the two-track final gate did not pass. The decision is to retain `model_candidate_v2`; no v3 model or submission CSV was exported, and hidden-set files and labels were not read.

The reproducible record is in `experiments/model_optimization_v2/` and `outputs/model_optimization_v2/`. Key artifacts are `protocol.json`, `method_registry.json`, `trial_ledger.jsonl`, `method_status.json`, `checkpoint_C1.json`, `checkpoint_C2.json`, `checkpoint_C3.json`, `composition_results.json`, `freeze_manifest.json`, and `final_benchmark_results.json`. The saved per-home final predictions are `predictions/final_benchmark/off_market.csv` and `predictions/final_benchmark/list_aware.csv`. The final state is 140/180 fits, 159.38/240 minutes, and `final_benchmark_completed`.

The benchmark runner needed a track-specific lookup for the frozen C0 metrics and an explicit import of the evaluation scoring function; both were corrected and covered by a focused regression test. Verification passed: 20 focused v2 tests, then the full repository suite with `OMP_NUM_THREADS=1` (62 passed in 82.36 seconds). The thread limit matters for the end-to-end starter validation test, which stalled without it.

### Next-stage goal (active)

The next research goal is active. It begins with a full audit of prior v1/v2 experiment records to identify genuinely unused calendar months, then pre-registers fresh time-based folds before any new fitting. It will investigate the generalization gap where list-aware M13 passed the four-month development gate but worsened on the retrospective benchmark, and why off-market experiments produced no stable gain. `model_candidate_v2` remains the frozen submission candidate. If there are not enough untouched folds, the next stage stops with that evidence rather than reusing the six-month retrospective benchmark as a new holdout. Another model candidate is exported only if both tracks pass the predeclared paired-home and month-stability gates.


## Fresh-fold model-research result — 2026-10-19

The fresh-fold stage completed using the locked model_candidate_v2 configurations as the parent baseline. The initial fit window was raised from 180 seconds to 300 seconds only for retrying missing or timed-out fold components; candidate definitions, data, features, targets, blends, fold dates, acceptance thresholds, and four-thread ceiling stayed fixed. All 18 missing/expired components recovered, 18 other components were cache hits, with zero unresolved failures or skipped folds. The experiment ledger contains 49 counted fit starts across the original run and recovery. From run_state.json start (2026-10-19 03:44:36 UTC) through recovery completion (05:40:10 UTC), wall time was 115.57 minutes, within the 180-minute development limit and preserving the 60-minute final reserve. The first run had 18 completed fits, 12 timeouts, and one abandoned launcher attempt; recovery completed 18 fits and reused 18 cached components. No final-month or hidden-test labels were read.

The month audit confirmed four unused, chronological development folds. Each fold trains strictly on sales before its held-out month:

| Held-out month | Training rows | Evaluation homes |
|---|---:|---:|
| 2024-10 | 112,637 | 2,451 |
| 2024-12 | 117,295 | 1,345 |
| 2025-01 | 118,640 | 1,339 |
| 2025-03 | 121,488 | 1,931 |

The combined paired evaluation contains 7,066 homes for each track. Fresh-fold baseline scores are off-market MdAPE 9.1745% / MAPE 11.7867%, and list-aware MdAPE 4.4726% / MAPE 5.5769%. These are scores on the fresh 2024-10 through 2025-03 development folds; do not substitute the older retrospective December 2025–May 2026 benchmark figures.

Three frozen hypotheses were evaluated per track: LightGBM quantile median (Q50), CatBoost MAE, and LightGBM Gamma trained on dollar price. Candidate promotion required at least 1% pooled MdAPE improvement, gains in at least three of four months, no monthly regression above 5%, no MAPE regression above 1%, and a paired-home bootstrap 95% interval whose upper endpoint was below zero. Results below report relative MdAPE change (positive is better), relative MAPE change (negative is better), count of improved months, and the 95% CI for candidate-minus-baseline MdAPE in percentage points.

| Track / candidate | MdAPE improvement | MAPE change | Months improved | Paired 95% CI (pp) | Result |
|---|---:|---:|---:|---:|---|
| Off-market Q50 | 0.57% | +0.83% | 1/4 | -0.211 to +0.156 | Fail: gain too small, only one month improved, CI crosses zero |
| Off-market CatBoost MAE | -15.12% | +13.13% | 0/4 | +1.148 to +1.633 | Fail: materially worse on every development month |
| Off-market Gamma price | 1.21% | +0.97% | 2/4 | -0.237 to +0.055 | Fail: only two months improved and CI crosses zero |
| List-aware Q50 | 1.54% | -0.84% | 4/4 | -0.122 to +0.023 | Closest candidate; passes every gate except the bootstrap upper bound |
| List-aware CatBoost MAE | 1.41% | -0.26% | 3/4 | -0.144 to +0.035 | Passes point-estimate gates; fails the bootstrap upper bound |
| List-aware Gamma price | -0.43% | +2.25% | 0/4 | -0.033 to +0.117 | Fail: pooled MdAPE and MAPE worsened |

Neither track has a candidate passing every development gate, so the freeze manifest selects no challenger and keeps final_months_unlocked=false. The reserved 2025-05 and 2025-08 final folds remain unopened. No final benchmark was run, no v3 bundle or official-format CSV was exported, and model_candidate_v2 remains the candidate. This is a valid negative selection result; the positive list-aware point estimates are not sufficient evidence for a promotion because their confidence intervals include zero.

Reproduction and review artifacts are in experiments/model_research_fresh/ and outputs/model_research_fresh/. Key files: protocol.json fixes the splits and gates; month_audit.json records the data/fold audit; candidates.py defines the three hypotheses; trial_ledger.jsonl records every fit, cache hit, timeout, and retry; development_results_pre_recovery.json and freeze_manifest_pre_recovery.json preserve the first run; development_recovery_results.json contains the complete recovery evaluation; development_results.json and freeze_manifest.json contain the final development decision; prediction files are under predictions/development/. Recovery implementation is experiments/model_research_fresh/recover_slow_fits.py. Focused tests are in tests/test_fresh_fold_research.py.

The strongest next research lead is list-aware Q50: it improved all four fresh months, reduced MdAPE by 1.54%, and improved MAPE by 0.84%, but the paired-home interval upper endpoint was +0.023 pp. List-aware CatBoost has a similar point gain but weaker month consistency and an upper endpoint of +0.035 pp. Off-market Gamma's 1.21% pooled gain was inconsistent across months and its MAPE change was close to the allowed ceiling. A follow-up should first audit whether additional unused months can strengthen the development evidence, then examine calibration/ensemble or market-regime hypotheses with predeclared tests. Keep 2025-05 and 2025-08 reserved for one final check; do not open them unless both tracks pass development gates. Any future stage must preserve the fit/time caps and keep v2 unchanged until a candidate passes both tracks.


Validation completed after the recovery write: a separate Python CSV/NumPy calculation over the saved per-home pipeline predictions reproduced all 7,066-home baseline and candidate MdAPE/MAPE point estimates and monthly-improvement counts. Parent/candidate keys and sale prices aligned for every track, method, and held-out month. The month audit confirms unique sale IDs and zero missing or nonpositive sale prices in the allowed development data. The final audit and final results files are absent, and the manifest keeps the reserved folds locked. Verification passed: python -m compileall -q experiments/model_research_fresh; git diff --check; and python -m pytest -q tests/test_fresh_fold_research.py --basetemp=.pytest_cache/fresh-fold-basetemp (6 passed in 92.67 seconds). No full repository suite was run in this stage.

### Next-stage research objective (registered 2026-10-19)

The active next-stage objective is to secure a reproducible improvement over the locked v2 baseline on both tracks, using a wider, audited development set and methods selected from the measured error patterns. It is tracked as the current long-term goal in Codex. First inventory the remaining unused pre-final months and select at least six additional chronological folds across the 2022–2024 regimes, while retaining the current four-fold findings as already-used development evidence and keeping May/August 2025 unopened as the one-time final check. Then diagnose paired residuals by month, price band, property type, and geography. Test only a small, preregistered set of distinct hypotheses: a past-only OOF calibration for list-aware Q50; a constrained Q50/CatBoost blend with weights learned exclusively from inner chronological OOF predictions; and one off-market recency or market-regime correction supported by the residual audit. Do not repeat off-market CatBoost MAE or list-aware Gamma unchanged. Keep per-fold training strictly before the sale month, preserve row-level predictions and code/data/environment hashes, run one learner at a time with at most four threads, and stop at 240 minutes or 180 counted fits with a 60-minute final reserve. Require both tracks to improve pooled MdAPE by at least 1%, show improvement in at least 75% of development months, keep worst-month MdAPE regression at or below 5%, MAPE regression at or below 1%, and pass the paired-home bootstrap upper-bound test. Open the two reserved final months only after both tracks pass; export a v3 package only if both final checks pass. Otherwise keep v2 and report the strongest point estimates and remaining uncertainty.

## Next-stage research implementation checkpoint at launch — 2026-10-19

The new research protocol is implemented in `experiments/model_research_next/`; detailed artifacts are under `outputs/model_research_next/`. Month audit scanned 3 historical experiment ledgers and 383 prediction CSVs with a month column. The union of prior evaluated months contains 17 months (2024-10 through 2026-05, with gaps); source data SHA-256 matches the frozen v2 hash `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`. The audit read only `ml_num` and `date_sold` before 2024-10-01; it did not read sale prices, 2025 final labels, or hidden files. Eight unused 2022–2024 folds provide 18,045 evaluation homes. The frozen plan uses 2022-02 (3,201 homes) only as a past-OOF seed, 2022-03 and 2022-10 for screening, and six confirmation folds from 2023-03 through 2024-09. Exact train/evaluation counts are in `outputs/model_research_next/month_audit.json` and `experiments/model_research_next/protocol.json`. 2025-05 and 2025-08 remain locked for one final check after both tracks pass confirmation.

A reproducible paired-residual report was generated from the previously saved four-fold predictions, pairing on month and listing ID and verifying source sale month/price. Q50 reduced list-aware MdAPE from 4.4726% to 4.4039% (-1.54%), MAPE from 5.5769% to 5.5302% (-0.84%), and median signed error from +0.765% to +0.440%. Off-market Gamma reduced MdAPE from 9.1745% to 9.0636% (-1.21%) but increased MAPE by 0.97%; its gain differed by month and house type. Actual-price bands are marked post-hoc descriptive only. Reports and paired rows are in `outputs/model_research_next/residual_diagnostics/`.

The preregistered registry has 20 fixed method cards across OOF global/group/prediction-band calibration, OOF-selected blends, Q50 and CatBoost candidate branches, and recency weighting. Screening selects at most four cards per track; confirmation uses six new months and requires at least 5/6 improved months, at least 1% pooled MdAPE gain, no monthly regression above 5%, MAPE regression at most 1%, and a paired-home bootstrap upper bound below zero. The budget is 240 minutes/180 fit starts, with 180 minutes reserved for development, 60 for final acceptance, four threads maximum, and one learner process at a time.

Implemented paired residual diagnostics, strictly past-only OOF calibration with optional rolling windows and prediction-derived price bands, constrained OOF blend weights, an audited fold plan, a subprocess worker, fit ledger, cache fingerprints, and final-label lock. Focused verification passed: 10 tests in `tests/test_model_research_next.py`; `python -m compileall -q experiments/model_research_next`; read-only preflight passed with all 9 frozen fold counts matching. The development runner started at 2026-10-19 06:14 UTC. At this checkpoint it had 6 fit starts, 5 completed, and the list-aware Q50 component for the 2022-02 seed was running. No final or hidden labels had been read; no candidate result or v3 export decision exists yet.

### Runtime update — 2026-10-19 07:05 UTC

The 20-card screening stage finished on the frozen 2022-03 and 2022-10 screening folds. Preliminary (not confirmation) list-aware results: `LA07_C0_Q50_oof_blend` improved pooled MdAPE by 1.66% and MAPE by 0.91%; `LA01_Q50_raw` improved MdAPE by 1.53% and MAPE by 0.90%; the next two cards also improved both metrics by about 1.31–1.34% and 1.00–1.22%, respectively. Off-market's top four cards did not improve pooled MdAPE on the screen folds, so none is considered a demonstrated gain. The frozen rule still sends the top four per track into the six-fold confirmation stage; these ranks do not determine adoption.

At this update, confirmation is fitting the first new month, 2023-03. The fit ledger contains 61 starts and 60 finishes; the active job is list-aware CatBoost MAE. No fit failure has been recorded, and no final labels have been opened. Final folds 2025-05 and 2025-08 remain locked. The current screen files are `outputs/model_research_next/screen_results.json` and `candidate_shortlist.json`; live fit-level provenance is in `fit_ledger.jsonl`.

### Next-stage 20-method research result — 2026-10-19

The preregistered development search completed in 100.512 minutes with 126/180 fit starts; all 126 recorded fits completed successfully. It evaluated 20 distinct fixed method cards on two screening months and confirmed the top four per track on six previously unused months (2023-03, 2023-07, 2023-11, 2024-03, 2024-06, 2024-09), totaling 12,372 paired homes. Data SHA-256 remained `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`; frozen v2 code/configuration fingerprints matched the run manifest. No new candidate passed every development gate, so the protocol correctly kept both reserved 2025 folds locked. `final_labels_opened=false`, `final_months_unlocked=false`, and `final_results.json` does not exist. `model_candidate_v2` remains the candidate; no v3 or submission CSV was exported.

| Track and confirmed card | Baseline → candidate MdAPE | Relative MdAPE gain | MAPE change | Months improved | Paired-home 95% CI for MdAPE delta (pp) | Decision |
|---|---:|---:|---:|---:|---:|---|
| Off-market `OM08_C0_recency_oof_blend` | 9.3161% → 9.3341% | -0.19% | +0.12% | 0/6 | [-0.033, +0.053] | Fail |
| Off-market `OM07_C0_Q50_oof_blend` | 9.3161% → 9.3229% | -0.07% | -0.51% | 2/6 | [-0.081, +0.073] | Fail |
| Off-market `OM01_Q50_raw` | 9.3161% → 9.3565% | -0.43% | +0.70% | 1/6 | [-0.112, +0.196] | Fail |
| Off-market `OM09_recency_weighted` | 9.3161% → 9.6298% | -3.37% | +2.46% | 0/6 | [+0.163, +0.463] | Fail |
| List-aware `LA07_C0_Q50_oof_blend` | 5.0906% → 5.0757% | +0.29% | -0.28% | 4/6 | [-0.062, +0.034] | Fail |
| List-aware `LA01_Q50_raw` | 5.0906% → 5.0619% | +0.56% | -0.13% | 4/6 | [-0.091, +0.031] | Fail |
| List-aware `LA09_CB_MAE_raw` | 5.0906% → 5.0786% | +0.24% | -0.06% | 3/6 | [-0.091, +0.063] | Fail |
| List-aware `LA08_Q50_CB_oof_blend` | 5.0906% → 5.0529% | +0.74% | -0.19% | 5/6 | [-0.097, +0.048] | Fail |

Positive MdAPE gain means error decreased; negative means it increased. The strongest list-aware result (`LA08`) was directionally promising but fell short of the required 1% pooled MdAPE reduction, and its bootstrap upper endpoint was above zero. The best off-market point estimate still increased MdAPE by 0.07%; no off-market candidate met the month-consistency or uncertainty requirements. Thus the screen-stage gains did not generalize strongly enough to establish a two-track upgrade. The result is a completed negative development finding, not evidence that v3 is better.

Reproducible details are in `outputs/model_research_next/confirmation_results.json`, `freeze_manifest.json`, `screen_results.json`, `candidate_shortlist.json`, `fit_ledger.jsonl`, and `method_status.json`; per-home predictions are under `outputs/model_research_next/predictions/`. Verification after completion passed: `10 passed` in `tests/test_model_research_next.py`, `python -m compileall -q experiments/model_research_next`, and a manifest check confirming 126/126 completed fits, no qualified track, and the final-label lock. The Codex research goal is complete under its stop-and-report rule; any further search needs a separately scoped next-stage objective.

## Team handoff and repository publication — 2026-10-19

Created `TEAM_PROGRESS.md` as a standalone teammate handoff. It consolidates the baseline and v2 selection result, remarks audit, three-checkpoint search, four fresh chronological folds, and 20-card confirmation; it distinguishes the retrospective six-month benchmark from unused chronological development folds and records that no v3 candidate passed promotion. `README.md` now links to the handoff and labels `release_candidate/` as the archived v1 snapshot. The repository publication includes the experiment code, plans, focused tests, candidate package, current submission CSVs, and generated experiment evidence under `outputs/`, while leaving the raw `data/` directory and local environments out of version control. No experiment result or candidate decision changed in this handoff step.
