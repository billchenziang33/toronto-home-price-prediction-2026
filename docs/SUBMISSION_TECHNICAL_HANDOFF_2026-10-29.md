# SSU Data Dash 技术交付与当前基准（2026-10-29）

## 当前结论

官方提交入口现选择 **Off-market R52** 和 **List-aware R97**。选择依据是比赛手册规定的六个月标准 walk-forward 窗口中各自最低的 MdAPE。当前分数是回顾性开发验证，不是独立 holdout，也不是隐藏测试成绩。该窗口曾用于先前模型开发，重复比较可能造成选择偏差。

最终预测文件已用这两个路由重新生成，并通过官方 `python run.py check`：

| 赛道 | 当前路由 | CSV 行数 | 六个月验证 MdAPE | 六个月验证 MAPE |
|---|---:|---:|---:|---:|
| Off-market | R52 | 3,974 | 7.0682% | 9.5223% |
| List-aware | R97 | 4,064 | 3.9396% | 5.0294% |

验证分数各基于 10,069 个 walk-forward 预测。不得把这些分数称作官方隐藏测试准确率。

## 当前模型流程

### Off-market R52

- 模型不读取 list price；当前官方 Off-market 测试输入也不含该字段。
- R22 与 R32 是 LightGBM 特征路线，包含基础预测和 Detached 专家预测；路线使用提供的挂牌描述，经训练折内 TF-IDF 形成特征。
- R48/R50 根据严格早于预测月份的安全 OOF 结果，按全局、房型及房型×价格组选择逐房混合权重。R52 再依据过去 OOF 记录自适应组合 R48 与 R50。
- R151 在 R52 上追加空间残差校正。它在八个月研究集上曾取得更低 MdAPE，但在本次六个月标准验证中比 R52 高 **0.1682 个百分点**（R151 减 R52 的配对 95% 家庭级 bootstrap 区间为 +0.0467 至 +0.2350 个百分点），所以当前提交改用 R52。

### List-aware R97

- 该赛道可用 list price。R22 LightGBM 与冻结 HGBR 组合；Detached 房屋使用专门模型结果，再叠加对应的 HGBR 预测。
- 之后用 CatBoost 预测 log(sold price / base prediction) 残差。残差元模型只使用严格早于预测 cutoff 的安全 OOF 行，训练目标及推理修正均经过裁剪和收缩。
- 当前提交路由 R97 在六个月标准验证中的 MdAPE 低于 R35。

## 六个月标准验证结果

窗口为 2025-12 至 2026-05。每个月只用该月第一天以前已成交的房屋训练；输出价格按最终提交方式四舍五入到整数美元后计分。MdAPE 是主排名指标，MAPE 是平分时的次指标。

| Track | Route | Homes | MdAPE %（家庭级 95% CI） | MAPE %（家庭级 95% CI） | P90 APE % |
|---|---|---:|---:|---:|---:|
| Off-market | R22 | 10,069 | 7.1932 (6.9976–7.3416) | 9.6365 (9.4493–9.8250) | 20.6467 |
| Off-market | R52 | 10,069 | **7.0682 (6.9112–7.2727)** | **9.5223 (9.3429–9.7114)** | 20.2300 |
| Off-market | R151 | 10,069 | 7.2364 (7.0357–7.3971) | 9.5250 (9.3486–9.7202) | 20.1963 |
| List-aware | R35 | 10,069 | 4.0058 (3.9138–4.1001) | 5.0753 (4.9955–5.1605) | 10.7002 |
| List-aware | R97 | 10,069 | **3.9396 (3.8650–4.0388)** | **5.0294 (4.9492–5.1146)** | 10.5750 |

R52 相对官方比较路线 R22 的 MdAPE 点估计低 0.1250 个百分点，配对家庭级 95% 区间为 −0.2072 至 +0.0129，包含零；因此这次比较未确认出统计上明确的改善。R97 相对 R35 的 MdAPE 点估计低 0.0661 个百分点，区间为 −0.1015 至 −0.0293。区间以房屋为抽样单位，没有涵盖同月共同市场冲击，也没有校正重复试验带来的选择偏差。

手册给出的 starter walk-forward 基线为 Off-market MdAPE 20.8% / MAPE 27.4%，List-aware 7.3% / 8.1%。以这些同窗基线作点估计比较，R52 的 MdAPE 低 13.73 个百分点，R97 低 3.36 个百分点；这只是内部回顾验证对照，不等于隐藏测试得分或官方准确度分。

误差分析显示，Off-market 的误差随成交价上升：最高价格十分位 MdAPE 为 13.34%（n=1,007）；Detached 为 9.59%（n=3,123）。List-aware 最高误差集中在第 9、10 价格十分位，MdAPE 分别为 5.46% 和 5.29%（各 n=1,007）；Semi-Detached 为 5.49%（n=1,034）。这提示后续改进可优先检查高价房与 Detached/Semi-Detached 子群。完整按月、房型、价格十分位和社区的结果见 [`../outputs/error_analysis.csv`](../outputs/error_analysis.csv)。

## 可复现记录

- 验证 run：`validation_20261029T164023Z_cff43dc1`。
- 最终预测 run：`submission_20261029T212456Z_056c5580`；独占训练 cutoff 为 `2026-06-01`，训练数据 150,000 行。
- 训练数据 SHA-256：`18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`。
- Off-market 测试输入 SHA-256：`d078d1bd652550a8bf33bbaa42af21e9a7ea6c42dfd53c57c083e5597165b1ad`；List-aware：`ef1b82d16d547199b93a4e1a27d1ff3946a327854c320956a32f437b81469b35`。
- Off-market CSV（统一 LF 换行）SHA-256：`bcb7ff2ac262161a7f3726621259c1ea1abfa6eaaad09191137cd1cb1f072c3f`；List-aware CSV：`c10e5b1229106f2aa3a368b18a9da0f3dde09cf4531cd37f4f59ebf840a390fd`。
- 预测 run 的 8/8 learner fits 均有完成记录；manifest 中的源文件 hash 与当前代码一致。训练及测试标签访问标记均为 `false`。运行清单里的机器本地路径已改为仓库相对路径。
- `.gitattributes` 将 CSV/JSONL 固定为 LF；文本文件 SHA 按 CRLF→LF 规范化，因此 GitHub、Windows 与 Linux checkout 对同一内容得到相同 hash。预测后只清理了 `run.py` 报告输出行的行尾空格，并同步刷新 manifest 的文件指纹；没有改动模型或预测逻辑。
- 六个月验证 manifest 的 run-time 模型记录早于 Off-market 路由从 R151 切换为 R52；`validate` 当时仍为五条路线都写出逐房预测，故所列 R52 分数未受默认选择字段影响。提交 manifest 则记录了最后选择的 R52/R97。
- 跨环境复跑通过：在 Git commit `9e50e8d` 的新隔离 checkout 中，使用另一套现有 `.venv`（Python 3.13.1，NumPy 2.2.6、pandas 2.3.3、scikit-learn 1.7.2、LightGBM 4.7.0、CatBoost 1.2.10、SciPy 1.18.1）运行 `predict`。复跑 `submission_20261029T214846Z_2c8f99b4` 的 8/8 fits 完成；两个 CSV 与原提交版本逐字节相同，SHA-256 分别为 `bcb7ff2ac262161a7f3726621259c1ea1abfa6eaaad09191137cd1cb1f072c3f` 和 `c10e5b1229106f2aa3a368b18a9da0f3dde09cf4531cd37f4f59ebf840a390fd`。隔离 checkout 的 `run.py check` 两项均通过；36 个源文件 hash 无差异。两个本机环境均通过 `pip check`；这是新 checkout 与第二套既有环境的复跑，不是从空目录重新安装依赖。
- 官方检查通过：`submission_off_market.csv: OK`，`submission_list_aware.csv: OK`。额外检查确认列为 `ml_num,predicted_price`、ID 无重复、价格均为正、CSV hash 与 manifest 一致。
- 路由切换后的聚焦集成测试通过：6 项通过。

结果、清单与误差图：[`../reports/validation_report.md`](../reports/validation_report.md)、[`../reports/validation_report.json`](../reports/validation_report.json)、[`../outputs/validation_run_manifest.json`](../outputs/validation_run_manifest.json)、[`../submissions/submission_run_manifest.json`](../submissions/submission_run_manifest.json)、[`../figures/error_by_price_range.png`](../figures/error_by_price_range.png)。

## 最终比赛要交什么

手册要求通过提交表单交 **一份不超过四页的打字 PDF 报告**。PDF 要链接：

1. 不超过五分钟的 YouTube 视频（unlisted 可以，private 不可以）；
2. 一个公开或与组织方共享的 GitHub 仓库，仓库内包含源代码和两份 CSV：`submissions/submission_off_market.csv`、`submissions/submission_list_aware.csv`。

报告应写全体队员完整姓名、两赛道 walk-forward MdAPE，以及所用额外数据源。当前项目没有加入外部数据。独立的 slide deck 不在手册列出的必交清单里，可在视频中用作讲解材料。表单链接：[比赛提交表单](https://forms.gle/QePTCKF9WEUjqDe17)。截止时间为 2026-11-08 23:59 EST。
