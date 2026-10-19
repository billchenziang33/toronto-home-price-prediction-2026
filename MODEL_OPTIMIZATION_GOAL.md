# 已完成目标记录：两赛道模型与预测方法优化

状态：2026-10-18 已完成。共 66/72 次真实数据 fit；双轨新候选通过预设回顾性验证门槛。当前冻结候选仍可直接回退。

实测：off-market MdAPE 从 10.5956% 降到 9.1733%，MAPE 从 13.4001% 降到 11.9803%；list-aware MdAPE 从 4.3596% 降到 4.2357%，MAPE 从 5.5651% 降到 5.3575%。两轨都是 6/6 月改善且 paired home-bootstrap 95% 区间上界小于 0。验证月份已在早期工作中查看，不能视作独立新测试。导出的模型和两份通过检查的 CSV 在 `model_candidate_v2/`。

两轨最终选择：off-market 使用 LightGBM log-price，preset 6、权重 1.00；list-aware 使用 LightGBM log(sold/list)，preset 5，与冻结 HGBR 按 0.75/0.25 融合。完整实验过程见 [执行计划](docs/superpowers/plans/2026-10-18-two-track-model-optimization.md) 和 [summary.md](summary.md)。

**下一阶段推荐目标：** 让报告、六页 slides、视频讲稿、模型源代码和两份候选 CSV 全部匹配 `model_candidate_v2`；重新核验竞赛数字、PDF 页数、slides 可读性和视频时长；待团队提供实际视频链接后，准备一致的公开仓库版本和提交信息。视频链接需要团队录制后提供；发布和提交表单是外部操作，应等用户在那一阶段授权后执行。

下面四段保留本轮实际执行的原始目标文字，供审计和复用：

基于 TorontoHomePriceChallenge 现有工作，开展一次有预算上限、可恢复的两赛道模型优化，争取同时降低 off-market 与 list-aware 的时间前推验证 MdAPE。以 release_candidate 中 commit 7cffe186bd9f0a2e8fc22f3897d52b3dbaaa8df8 的 10.60% / 4.36% 为冻结参照，原包保持可回退。算法不限于现有 HistGradientBoostingRegressor：优先公平比较当前模型、CatBoost、LightGBM，必要时用 XGBoost 替补；两轨可选择不同算法。只调各轨胜者的小参数集，并检验 off-market 的 L1 稳健损失、list-aware 的 log(sold_price/list_price) 目标，以及冻结模型与一个新模型的简单融合。所有结论须由本项目同房、同月、前推验证支持。

争取两轨 MdAPE 各相对下降至少 5%，约达到 ≤10.07% / ≤4.14%；该数值为研究目标，不是承诺。最低升级幅度为两轨各相对下降至少 2%，并通过逐月稳定性、MAPE、正确配对 bootstrap、数据权限、复现与 CSV 校验。只一轨改进就报告部分进展，没有可靠改进就保留原候选并给出完整负结果。

按上述完整计划推进：先核验基准与环境，再在固定的 2025-06 / 2025-09 / 2025-11 开发窗口做算法筛选、小范围调参、目标变换和融合选权重；锁定配置后，每轨一个候选对 2025-12 至 2026-05 做一次完整复核。已有六个月已被多轮查看，不能宣称为全新的独立确认集；隐藏测试只生成最终预测，不参与选择。默认不用外部数据或重新开展旧 remarks/SVD 宽搜索。

单轮最多 180 分钟墙钟时间或 72 次模型 fit，先到者停止；按 pilot 耗时预留最终核验预算，最多 4 线程且一次只训练一个模型。逐 fit 保存配置、数据/代码指纹、截止日、预测、指标、耗时、失败与检查点，精确缓存命中就不重算。交付开发和六个月对照、逐月/分组误差、完整日志、赢家配置；两轨通过时在独立 model_candidate_v2 中导出可加载模型、精确依赖、代码和两份合格 CSV。更新 summary.md 与桌面归档后结束本阶段。未达到期望分数也要如实交付并停止，不无限调参；不在此阶段重做 PDF/slides、录视频、发布仓库或提交表单。

完整步骤、固定参数、预算与采用门槛见 [执行计划](docs/superpowers/plans/2026-10-18-two-track-model-optimization.md)。版本决策和历史见 [summary.md](summary.md)。
