# 技术文献参考清单

> 面向本项目的国内外相关文献梳理。每条文献标注「对应本项目哪一层 / 支撑什么论断」，
> 便于写论文、做答辩或写技术报告时直接定位，而不是堆一份与代码无关的书目。
>
> 检索与核对时间：2026-11-23。所有条目的卷期、页码、DOI 均经检索核对；
> 凡未能核实的字段一律留空或标注，不做推测填充。

## 使用说明（先看这一节）

文献能解决三类问题，也**不能**解决一类问题，必须分清：

**能解决**

1. **方法出处**：你用的 YOLOv8 / ByteTrack / Isolation Forest / GAE / Louvain 都不是自创，
   引用原始论文是学术规范，也是答辩时"为什么选这个方法"的第一层答案。
2. **参数与默认值的依据**：例如 GAE 用 GCN 编码器、Louvain 的模块度目标、IF 的子采样规模，
   原始论文里都有取值建议。
3. **已知缺陷的权威背书**：本项目若干"已知限制"（如 Louvain 会产生内部不连通的社区、
   单类学习对未知异常无法保证召回）在文献中早有结论。**引用它们，比在文档里写"我们发现"
   更有说服力**，也能说明这不是实现 bug 而是方法本身的性质。

**不能解决**

4. **指标 7 / 9 / 14 / 15 的召回率、AUC、精确率**。这四项缺的是**现场标注数据**，
   不是方法参考。任何文献都无法替代"人工标注一批高风险车/违规车"这件事。
   引用文献不会让 🔴 变成 ✅，这一点在 [`ACCEPTANCE_REPORT.md`](ACCEPTANCE_REPORT.md) 里已如实记录。

---

## 一、异常检测基础（对应第二层、第四层）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [1] | CHANDOLA V, BANERJEE A, KUMAR V. Anomaly detection: A survey[J]. ACM Computing Surveys, 2009, 41(3): Article 15, 1-58. DOI: 10.1145/1541880.1541882 | 第二层方法选型的总体框架；支撑"无监督 / 单类学习适用于无标注场景"这一前提 |
| [2] | LIU F T, TING K M, ZHOU Z H. Isolation Forest[C]//Proceedings of the 8th IEEE International Conference on Data Mining (ICDM). Pisa: IEEE, 2008: 413-422. DOI: 10.1109/ICDM.2008.17 | `src/anomaly/isolation_forest.py` 的算法出处；iForest 的线性复杂度与低内存特性是"边缘部署"论断的依据 |
| [3] | XU J, WU H, WANG J, LONG M. Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy[C]//International Conference on Learning Representations (ICLR). 2022. | 第二层"用注意力偏离刻画异常"的理论参照；可对比说明本项目采用**预测残差**而非关联差异的取舍 |
| [4] | TULI S, CASALE G, JENNINGS N R. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data[J]. Proceedings of the VLDB Endowment, 2022, 15(6): 1201-1214. DOI: 10.14778/3514061.3514067 | Transformer 用于异常检测的代表工作；其"两遍重构放大偏差"思路可解释本项目为何保留残差作为特征 |

## 二、目标检测与多目标跟踪（对应第一层）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [5] | REDMON J, DIVVALA S, GIRSHICK R, FARHADI A. You Only Look Once: Unified, Real-Time Object Detection[C]//IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016: 779-788. DOI: 10.1109/CVPR.2016.91 | YOLO 系列的源头；单阶段检测"速度优先"的取舍依据 |
| [6] | ZHANG Y, SUN P, JIANG Y, et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box[C]//Computer Vision – ECCV 2022. LNCS 13682. Cham: Springer, 2022: 1-21. DOI: 10.1007/978-3-031-20047-2_1 | `src/tracking/bytetrack_tracker.py` 的算法出处；"低分框二次关联"直接对应本项目"ID 保持 / 丢失重连"的实现目标 |
| [7] | 金沙沙, 龙伟, 胡灵犀, 王天宇, 潘华, 蒋林华. 多目标检测与跟踪算法在智能交通监控系统中的研究进展[J]. 控制与决策, 2023, 38(4): 890-901. DOI: 10.13195/j.kzyjc.2021.1763 | 国内智能交通场景下 TBD 范式的综述；可支撑"为什么在卡口场景选 tracking-by-detection 而非联合检测跟踪" |
| [8] | 邵延华, 张铎, 楚红雨, 等. 基于深度学习的 YOLO 目标检测综述[J]. 电子与信息学报, 2022, 44(10): 3697-3708. | 中文语境下 YOLO 系列演进与选型依据，适合作为国内文献支撑 |

## 三、车辆行为与轨迹建模（对应第一层特征体系、第二层）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [9] | 黄超, 胡志军, 徐勇, 王耀威. 基于视觉的车辆异常行为检测综述[J]. 模式识别与人工智能, 2020, 33(3): 234-248. DOI: 10.16451/j.cnki.issn1003-6059.202003005 | **本项目最贴合的中文综述**。其"行为表示特征 → 行为学习 → 行为建模"的三段式分类，与项目四层结构可逐段对应；文中指出的"异常定义不确切、缺乏标注、遮挡、实时性差"四类挑战，正是本项目 `ACCEPTANCE_REPORT.md` 中 🔴/🟡 项的成因 |
| [10] | VASWANI A, SHAZEER N, PARMAR N, et al. Attention Is All You Need[C]//Advances in Neural Information Processing Systems (NIPS). 2017. arXiv:1706.03762. | `src/anomaly/trajectory_transformer.py` 的架构出处 |

## 四、图表示学习与图异常（对应第三层、指标 11）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [11] | KIPF T N, WELLING M. Semi-Supervised Classification with Graph Convolutional Networks[C]//ICLR. 2017. arXiv:1609.02907. | GCN 编码器的出处；本项目 GAE 的编码器结构基础 |
| [12] | KIPF T N, WELLING M. Variational Graph Auto-Encoders[J]. arXiv preprint arXiv:1611.07308, 2016 (NIPS 2016 Workshop). | `src/graph/gae.py` 的算法出处；"重构误差即异常度"的立论依据 |
| [13] | DING K, LI J, BHANUSHALI R, LIU H. Deep Anomaly Detection on Attributed Networks[C]//SIAM International Conference on Data Mining (SDM). 2019. | 属性图异常检测的代表工作（DOMINANT）；本项目"特征驱动 GCN + 重构误差"路线与其同源，可作为指标 11 的对照基线 |
| [14] | 陈波冯, 李靖东, 卢兴见, 沙朝锋, 王晓玲, 张吉. 基于深度学习的图异常检测技术综述[J]. 计算机研究与发展, 2021(7): 1436-1455. | 国内图异常检测的权威综述；适合支撑第三层的方法分类与选型说明 |

## 五、社区发现（对应第三层、指标 12/13）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [15] | BLONDEL V D, GUILLAUME J L, LAMBIOTTE R, LEFEBVRE E. Fast unfolding of communities in large networks[J]. Journal of Statistical Mechanics: Theory and Experiment, 2008(10): P10008. | `src/graph/community.py` 的算法出处；模块度 Q 的定义与优化目标 |
| [16] | TRAAG V A, WALTMAN L, VAN ECK N J. From Louvain to Leiden: guaranteeing well-connected communities[J]. Scientific Reports, 2019, 9: 5233. | **重要的"已知限制"背书**。该文明确指出 Louvain 可能产出内部不连通的社区。本项目用的是 Louvain，若报告中出现"社区内节点连接稀疏"的现象，引此文说明这是方法的已知性质，而非实现缺陷；同时可作为升级到 Leiden 的正当理由 |

## 六、跨摄像头车辆重识别（对应第一层多机位、`stitch_cameras.py`）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [17] | LIU X, LIU W, MEI T, MA H. A Deep Learning-Based Approach to Progressive Vehicle Re-identification for Urban Surveillance[C]//ECCV 2016: 869-884. | 提出 VeRi-776 数据集与 PROVID 框架；"由近及远搜索"的空间约束思路对应本项目"拓扑时间窗 + ReID 相似度"的拼接策略 |
| [18] | LIU X, LIU W, MEI T, MA H. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance[J]. IEEE Transactions on Multimedia, 2018, 20(3): 645-658. | 多模态（视觉 + 车牌 + 时空）融合的完整论述；本项目车牌识别率仅 0.90% 时必须以时空约束为主，此文说明该退化的合理性 |
| [19] | 平灿, 李雷孝, 刘东江, 林浩, 史建平. 基于深度学习的车辆重识别研究进展[J]. 计算机工程与应用, 2025, 61(16): 16-37. DOI: 10.3778/j.issn.1002-8331.2409-0384 | 最新的中文 ReID 综述；其"表征学习 / 跨域学习"二分与"跨域适应性"讨论，可支撑本项目"换机位后性能下降"的风险提示 |

## 七、可解释性与证据链（对应第四层、指标 16）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [20] | RUDIN C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead[J]. Nature Machine Intelligence, 2019, 1(5): 206-215. DOI: 10.1038/s42256-019-0048-x | **本项目"输出排查线索而非违规判定"的方法论依据**。该文主张高风险决策应使用本身可解释的模型，而非事后解释黑箱。项目的加权线性融合 R=αA+βG+γT 与逐条证据链，正属于"结构上可解释"的一类，可作为设计选择的正当性论证 |

## 八、大模型报告生成与幻觉控制（对应第四层、指标 17）

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [21] | JI Z, LEE N, FRIESKE R, et al. Survey of Hallucination in Natural Language Generation[J]. ACM Computing Surveys, 2023, 55(12): 1-38. DOI: 10.1145/3571730 | 支撑"必须对 LLM 输出做数字/编号白名单校验 + 失败回退模板"的设计；幻觉分类体系可用于说明本项目约束的是哪一类幻觉（事实性虚构） |
| [22] | LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[C]//NeurIPS. 2020. arXiv:2005.11472. | 本项目把结构化证据作为上下文喂给本地模型的做法的理论出处；可解释为何"模板回退"仍是必要的兜底 |

## 九、国内应用背景（综保区 / 智能卡口）

> 本节为**行业实践类**文献，非核心期刊，适用于交代应用背景与业务合理性，
> 不宜作为算法依据引用。

| 编号 | 文献 | 对应本项目 |
|---|---|---|
| [23] | 李常铮, 李少杰, 张艳. 智能卡口技术运用于综合保税区信息化系统工程中的价值研究[J]. 科技创新导报, 2020, 17(5): 140-141. | 综保区卡口信息化的业务背景：卡口数据采集 → 平台比对 → 远程放行的既有流程 |
| [24] | 曹殷. 智能卡口技术在综合保税区信息化系统工程中的应用[J]. 城市建设理论研究(电子版), 2018, 8(30): 193. | 同上，可作为早期实践的补充佐证 |

---

## 十、文献与指标的对应关系（写报告时直接查这张表）

| 项目指标 | 可引文献 | 文献能证明什么 | 文献**不能**证明什么 |
|---|---|---|---|
| 指标 6（预测 3 秒位移 ≤2 m） | [10] | Transformer 用于序列预测的可行性 | 本项目的误差数值；且 3 秒不达标的原因是**轨迹中位仅 1.84 秒**（数据问题），与模型无关 |
| 指标 9（未知异常 AUC ≥0.85） | [1][2] | 单类学习在原理上覆盖未知类型 | AUC 数值本身。需要标注集，且当前合成注入不可用（A 与原样本逐位相同） |
| 指标 11（GAE 异常节点 ≥85%） | [11][12][13][14] | 重构误差作为异常度的有效性；度越界型异常可达 0.963 | 保度重连型 0.58-0.60 的失败。这一项是**同图自参照**导致，属方法性质，文献只能佐证不能解决 |
| 指标 12/13（团伙/社区） | [15][16] | 模块度优化的合理性与 Louvain 的已知缺陷 | 团伙真值。当前是"能力已在，缺回测数据" |
| 指标 14/15（R 的 AUC / 精确率） | [20] | 加权线性融合的可解释性优势 | 权重 α/β/γ 取 0.4/0.35/0.25 的**最优性**。这组权重是标称值，需标注数据才能验证，报告中不应暗示其经过优化 |
| 指标 16（证据链 100%） | [20] | 可解释模型在高风险场景的必要性 | 证据链的**业务有效性**（证据是否真的指向违规），这仍需人工核查 |
| 指标 17（LLM 事实还原） | [21][22] | 幻觉的成因与 RAG 的缓解作用 | 95% / 2% 这两个阈值。当前实测 5/5 通过，样本量过小，不足以支撑百分比结论 |
| 指标 19（20 路 1080P） | [6] | 跟踪环节的计算开销量级 | 本项目的 8.4 路实测值；瓶颈在检测与解码，不在跟踪 |

---

## 十一、检索方法与续查建议

本文献清单是"够用"而非"穷尽"。若需扩充（如写学位论文），建议按以下检索式续查，
并注意**只引用能拿到全文或至少能核到摘要的条目**。

**中文（CNKI / 万方 / 维普）**

```text
主题 = 车辆异常行为 AND 监控视频
主题 = 车辆重识别 AND 深度学习
主题 = 图异常检测 AND 属性网络
主题 = 卡口 AND (综合保税区 OR 海关特殊监管区)
篇关摘 = 多目标跟踪 AND 智能交通
```

**英文（Web of Science / Scopus / IEEE Xplore）**

```text
TS=("vehicle anomaly detection" OR "abnormal vehicle behavior") AND TS=surveillance
TS=("vehicle re-identification") AND TS=(survey OR review)
TS=("graph anomaly detection") AND TS=(attributed network)
TS=("multi-object tracking") AND TS=(traffic surveillance)
TS=("explainable anomaly detection") AND TS=(security OR surveillance)
```

**筛选优先级**

1. 近 5 年的**综述**先行——用于确定方法分类框架，避免遗漏主流路线；
2. 再找**原始方法论文**（本文 [2][6][12][15] 这一类），用于交代算法出处与参数依据；
3. 最后找**同场景应用论文**（智能交通、卡口、海关监管），用于论证工程可行性。

---

## 十二、诚实声明

1. 本清单**不含**任何未核实的条目。凡卷期、页码、DOI 未查到确切信息者，已省略而非补全。
2. [23][24] 两篇为行业实践类文章，期刊层次不高，仅适合作为业务背景引用，**不建议**作为
   算法或方法论的支撑文献。
3. 文献无法弥补本项目最核心的缺口：**现场标注数据**。指标 7、9、12、14、15 的 🔴 状态，
   根因是缺真值而非缺方法。补充文献不会改变这一结论。
4. 若在正式论文中引用，请以各期刊官网或 DOI 解析页显示的最终卷期页码为准——
   本清单的核对基于公开检索结果，个别条目可能存在网络出版与印刷出版页码不一致的情况。
