# 最新状态更新（2026-10-28，Toronto）

- 交接报告的数据 SHA-256 已修正为 `18cf87fa122b544018fccbc15040403b0fed37080c44a75d1b55deca96d2c4bf`，与 R151、R97 的运行清单及 campaign 协议一致；之前报告中的两个哈希字符串误写，不影响已保存的模型指标。修正范围仅为报告和此记录。
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

The Desktop archive mirror is `C:\Users\admin\Desktop\SSU Data Competition\TorontoHomePriceChallenge`; its final optimization sync is recorded in `MODEL_OPTIMIZATION_SYNC_2026-10-18_COMPLETE.json` with SHA-256 hashes for the copied files.

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

The verified 54-file candidate was copied to `C:\Users\admin\Desktop\SSU Data Competition\TorontoHomePriceChallenge\release_candidate\`. All 53 entries in its SHA-256 manifest passed on the Desktop copy. At the candidate handover, the Desktop-wide manifest verified 262 payload files. The later optimization-plan documentation adds two files (264 payload files total); the frozen candidate is unchanged and the archive manifest is refreshed after document synchronization.

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
