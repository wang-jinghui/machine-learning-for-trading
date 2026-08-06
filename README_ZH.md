# 机器学习交易 — 第3版

**构建、测试并部署基于机器学习的交易策略 — 从数据获取到实盘执行。**

本仓库是 [Stefan Jansen](https://www.linkedin.com/in/applied-ai/) 所著 [*Machine Learning for Trading, 3rd Edition*](https://amzn.to/4eigy2F) 的配套代码 —— 一次彻底的重新构建，围绕一个端到端的工作流程展开：如何定义研究构想，并将其迭代开发为你能够在真实市场中实际运行、持续运行的策略。

- [九个案例研究](https://www.ml4trading.io/case-studies/)贯穿全书27章，从原始数据到特征、模型、回测、成本、风险直至部署，完整展示整个工作流程。
- **生成式AI** 和 **自主智能体** 是本版新增内容，贯穿整个工作流程，将检索增强生成、知识图谱和多智能体系统引入金融研究。
- [配套网站](https://ml4trading.io) 提供 [112篇入门指南](https://ml4trading.io/primer/)、[61个智能体技能](https://ml4trading.io/skills/) 和 [六个生产级Python库](https://ml4trading.io/libraries/)，大幅简化工作流程的各个环节。

**从这里开始：[安装指南](docs/installation.md)** 引导你从一台全新的 Linux、Windows 或 macOS 机器搭建到可运行的 Notebook，包含所有前置依赖。简要版本见下方 [快速开始](#快速开始)。

> **免费读者指南：** 加入 [Navigate ML for Trading, 3rd Edition](https://maven.com/p/c6e0e7/navigate-ml-for-trading-3rd-edition)，
> 于 **2026年7月30日上午11:00（美东时间）** 参加30分钟的书本、案例研究、代码及配套资源导览。
> 查看所有最新的 [课程和工作坊](https://maven.com/stefan-jansen)；同期课程列在下方 [课程](#课程) 部分。

<p align="center">
  <a href="https://amzn.to/4eigy2F"><img src="assets/cover.png" width="45%" alt="Machine Learning for Trading, 3rd Edition"></a>
</p>

---

## 第三版新内容

全书贯穿一条主线：从数据基础设施和策略研究，跨越一条区分调优与评估的 *证据边界*，到部署与监控 —— 并通过反馈回路在策略边际收益递减时进行重训练、暂停或退役。

<p align="center">
  <img src="assets/workflow.png" width="90%" alt="ML4T工作流程：数据基础设施与策略研究、区分调优与评估的证据边界、以及包含重训练/暂停/退役反馈回路的部署">
</p>

前两版按技术逐一展开，第三版则将这一完整流程端到端地呈现 —— 并新增大量内容：

- **更广泛的模型工具箱**：从梯度提升（XGBoost、LightGBM、CatBoost）到深度时间序列架构（PatchTST、iTransformer、TSMixer、TCN、Mamba），以及更新的表格和隐因子模型（TabPFN、TabM、条件自编码器和监督自编码器）。
- **专属策略设计章节**：交易成本和风险管理现在各自成为独立完整章节（此前均不存在），与组合构建和策略综合一起，将原始信号转化为经过仓位管理、成本意识和风险感知的组合。
- **完整的生产轨道**：实盘交易系统（Interactive Brokers、Alpaca、QuantConnect）、MLOps与治理（漂移检测、安全发布、熔断机制、特征存储、实验追踪），以及策略 *运营* 的现实 —— 不仅仅是构建策略。
- **生成式AI**：基于SEC文件的检索增强生成、知识图谱与Graph RAG、以及自主多智能体研究系统。
- **因果机器学习**：Double ML、贝叶斯结构时间序列和因果发现，用于区分真实效应与虚假相关。
- **强化学习**：最优执行、带库存的做市和深度对冲。
- **合成金融数据**：TimeGAN、Tail-GAN、Sig-CWGAN和基于扩散的生成器，用于历史数据不足时的验证。

方法论的严谨性被视为首要议题。全书明确划分了探索与确认的界限 —— *证据边界* —— 全程使用滚动前进交叉验证，并以 Deflated Sharpe Ratio、Rademacher Anti-Serum 和 White's Reality Check 等工具应对悄然使大多数回测失效的多重检验和过拟合问题，同时使用共形预测提供诚实的不确定性估计。

数据层采用 **Polars** 进行快速、基于表达式的操作，每章均在 **可复现的 Docker 环境** 中运行，确保跨机器结果一致；PyTorch、LightGBM、Optuna 和 Plotly 构成了建模和可视化工具栈。

### 九个案例研究

第三版的结构核心是 **[九个案例研究](case_studies/)**，贯穿全书。ETF、加密货币永续合约、日内股票、期权、外汇、期货和股票因子面板各自通过 *同一* 流水线 —— 从原始数据和标签到特征、模型、回测、成本、风险叠加层，直至最终部署评估。一套严谨的流程应用于九个截然不同的市场，展示其在哪里有效、在哪里失效以及原因。

| 案例研究                                                              | 资产类别          | 频率     | 探索内容                                                                     |
|-------------------------------------------------------------------------|--------------------|----------|------------------------------------------------------------------------------|
| [ETF](case_studies/etfs/)                                              | 多资产ETF          | 日频     | 100只ETF的跨资产动量与均值回归                                               |
| [加密货币永续合约](case_studies/crypto_perps_funding/)                  | 加密货币           | 8小时    | 永续期货的资金费率套利                                                       |
| [NASDAQ-100](case_studies/nasdaq100_microstructure/)                    | 股票               | 15分钟   | 来自订单流和限价订单簿的日内微观结构信号                                     |
| [S&P 500股票+期权](case_studies/sp500_equity_option_analytics/)        | 股票+期权          | 日频     | 利用隐含波动率特征增强股票选择                                               |
| [美国公司特征](case_studies/us_firm_characteristics/)                   | 股票               | 月频     | 公司层面特征面板（规模、价值、动量、质量）                                   |
| [外汇货币对](case_studies/fx_pairs/)                                    | 外汇               | 日频     | 主要货币对的套息和动量                                                       |
| [CME期货](case_studies/cme_futures/)                                    | 期货               | 日频     | 商品和金融期货的期限结构和展期收益信号                                       |
| [S&P 500期权](case_studies/sp500_options/)                              | 期权               | 日频     | 纯期权策略（跨式、Delta对冲头寸）                                            |
| [美国股票](case_studies/us_equities_panel/)                             | 股票               | 日频     | 具有经典因子暴露的美国股票广泛横截面                                         |

### 112篇入门指南

为全书所涉及的每个概念提供免费讲解。每个部分链接到其完整列表；以下主题展示其范围：

- [基础](https://ml4trading.io/primer/)：8个主题，涵盖限价订单簿机制、双时间数据模型以及模拟器必须复现的典型事实。
- [研究设计与特征工程](https://ml4trading.io/primer/)：21个主题，包括因子研究中的多重检验、分数阶差分和金融序列的路径签名。
- [模型开发](https://ml4trading.io/primer/)：22个主题，包括正则化几何、金融中的共形预测和双重机器学习的机制。
- [策略实施](https://ml4trading.io/primer/)：27个主题，从缩胀夏普比率、层次风险平价到 Almgren-Chriss 最优执行。
- [高级AI](https://ml4trading.io/primer/)：8个主题，如马尔可夫决策过程、策略梯度定理和事件预测的适当评分规则。
- [生产](https://ml4trading.io/primer/)：2个主题，冠军-挑战者评估和特征存储的训练-服务偏差。
- [跨领域概念](https://ml4trading.io/primer/)：24个构建模块，在各章中被引用，例如动量与均值回归、偏差-方差权衡和滚动前进验证。

### 61个智能体技能

可复用的、带防护措施的编码智能体任务，每个都内置了防止前视偏差、数据泄露和多重检验错误的防御。以下技能展示其范围：

- [概念](https://ml4trading.io/skills/)：10个技能，包括前视偏差、数据泄露和信息系数。
- [数据获取](https://ml4trading.io/skills/)：7个技能，涵盖数据获取、Bar构建和数据验证。
- [特征工程](https://ml4trading.io/skills/)：10个技能，包括特征计算、三屏障标签和特征选择。
- [评估与验证](https://ml4trading.io/skills/)：8个技能，从滚动前进交叉验证到清除-禁运和缩胀夏普比率。
- [回测](https://ml4trading.io/skills/)：5个技能，如运行回测、成本模型和报告表。
- [组合管理](https://ml4trading.io/skills/)：5个技能，包括仓位管理、风险指标和熔断开关。
- [基础设施](https://ml4trading.io/skills/)：4个技能，如规范模式、注册系统和Polars模式。
- [工作流](https://ml4trading.io/skills/)：5个技能，涵盖因子研究、模型验证和生产就绪。
- [生产](https://ml4trading.io/skills/)：2个技能，实盘交易和监控告警。
- [高级AI](https://ml4trading.io/skills/)：5个技能，涵盖研究算子、智能体记忆、预测、治理和RAG评估。

### 课程

在 [Maven](https://maven.com/stefan-jansen) 上提供的 [同期课程](https://ml4trading.io/courses/)，实时讲解书中内容并提供直接反馈：

- [Machine Learning for Trading: From Research to Production](https://maven.com/stefan-jansen/research-to-production)：将研究构想一路推进到已部署、已监控的策略。
- [Engineering a Multi-Agent Forecasting System](https://maven.com/stefan-jansen/agent-engineering)：设计可审计的多智能体系统用于金融研究。

每期课程按计划开设班次；以上链接始终指向下一期，可报名或加入候补名单。*在同期课程之间保持关注 [**Insights** 通讯](https://insights.ml4trading.io/)。*

---

## ML4T 库

Notebook 基于六个生产级 Python 包构建，每个包可独立使用 —— 对应工作流程的每个阶段：

| 库                                                            | 阶段     | 功能描述                                                     |
|---------------------------------------------------------------|----------|--------------------------------------------------------------|
| [`ml4t-data`](https://ml4trading.io/docs/data/)               | 数据     | 统一接口从19+数据提供商获取市场数据                          |
| [`ml4t-engineer`](https://ml4trading.io/docs/engineer/)       | 信号     | 特征、标签、替代Bar和防泄漏数据集准备                        |
| [`ml4t-models`](https://ml4trading.io/docs/models/)           | 模型     | 金融原生隐因子、SDF、直接预测和组合学习                      |
| [`ml4t-diagnostic`](https://ml4trading.io/docs/diagnostic/)   | 评估     | 特征验证、策略诊断和 Deflated Sharpe Ratio                   |
| [`ml4t-backtest`](https://ml4trading.io/docs/backtest/)       | 策略     | 带真实执行的事件驱动回测                                     |
| [`ml4t-live`](https://ml4trading.io/docs/live/)               | 部署     | 集成券商的实盘交易                                           |

---

一个引言章和一个总结章首尾呼应，中间是六个按工作流程排列的部分。每个章节标题链接到其指南；Notebook 按目录逐步完善。

## 引言

### [1. 流程即优势](01_process_is_edge/)

为什么流程纪律胜过模型复杂性。将 ML4T 工作流程作为研究到生产的系统进行介绍，涵盖因子收益和宏观指标的机制检测，以及区分探索与确认的证据边界。

## 第一部分 — 金融数据（第2-5章）

全书其余部分所依赖的市场、工具和基础设施：数据源分类、原始交易所消息转化为可用于特征的Bar、时间点基本面数据，以及用于稳健验证的合成历史数据。

### [2. 金融数据宇宙](02_financial_data_universe/)

市场、基本面和替代数据的分类。调查八种资产类别，量化生存偏差，基准测试存储格式（Parquet、DuckDB、kdb+、TimescaleDB），并建立全书使用的数据质量框架。

### [3. 市场微观结构](03_market_microstructure/)

从原始交易所消息到可用于特征的Bar。解析 NASDAQ ITCH，从多个数据源重建限价订单簿，验证 Lee-Ready 交易分类，并比较Bar采样方法 —— 美元Bar提供最佳的收益正态性。

### [4. 基本面与替代数据](04_fundamental_alternative_data/)

SEC EDGAR 文件的时间点流水线、跨标识符系统的实体解析、宏观和商品基本面，以及替代数据评估 —— 包括链上加密货币基本面和预测市场（Kalshi、Polymarket）。

### [5. 合成金融数据](05_synthetic_data/)

生成替代市场历史用于稳健验证。实现 TimeGAN、Tail-GAN、Sig-CWGAN、Diffusion-TS 和基于 LLM 的表格生成，通过保真度-效用-隐私框架进行评估。

## 第二部分 — 研究设计与特征工程（第6-10章）

定义交易问题，然后将数据转化为模型可用的信号：研究设计、标签、特征以及决定任何模型能学到什么的评估。

### [6. 策略研究框架](06_strategy_definition/)

在构建模型之前定义交易博弈：宇宙规则、决策时间表、成本模型、评估协议和运行日志。介绍九个案例研究和锚定第7-20章的滚动前进交叉验证纪律。

### [7. 定义学习任务](07_defining_the_learning_task/)

标签工程（前向收益、三屏障、趋势扫描）、单变量特征评估（信息系数、分位数分析、可行性筛选）、多重检验控制（BH-FDR、Deflated Sharpe Ratio）和因果合理性检查。

### [8. 金融特征工程](08_financial_features/)

来自价格数据的五大特征族（动量、反转、波动率、流动性、微观结构）、结构和跨工具特征（收益率曲线、期限结构、相对价值）、上下文特征（宏观机制、日历、情绪），以及带稳健性测试的特征选择。

### [9. 基于模型的特征提取](09_model_based_features/)

来自拟合模型的特征：平稳性诊断、卡尔曼滤波器、傅里叶和小波谱特征、GARCH 波动率以及 HMM 机制概率 —— 全程确保时间点正确性。

### [10. 文本特征工程](10_text_feature_engineering/)

从词袋到 Transformer：TF-IDF、Word2Vec 和 GloVe 嵌入、LSTM 序列模型、FinBERT 情绪分析、金融 NER 微调和新闻收益信号构建。

## 第三部分 — 模型开发（第11-15章）

五个模型族应用于相同的九个案例研究，每个都在基线模型之上构建。

### [11. 机器学习流水线](11_ml_pipeline/)

正则化线性模型（Ridge、LASSO、Elastic Net）作为后续每个模型必须超越的基线。逻辑回归用于方向预测、SHAP 可解释性、共形预测用于不确定性估计，以及跨九个数据集的比较。

### [12. 梯度提升与高级表格模型](12_gradient_boosting/)

XGBoost、LightGBM 和 CatBoost 配合 Optuna 多目标调优，以及深度学习表格替代方案（TabPFN、TabM）。TreeSHAP 可解释性和跨数据集结果，梯度提升在大多数案例研究中是最强的表格模型。

### [13. 时间序列深度学习](13_dl_time_series/)

LSTM、N-BEATS、Transformer（PatchTST、iTransformer、TFT）、TSMixer、TCN 和 Mamba，结合 LTSF-Linear 讨论。一个实践者选择框架和跨数据集证据，说明深度学习何时有效、何时简单模型即可。

### [14. 隐因子模型](14_latent_factors/)

PCA 特征组合、带时变载荷的 IPCA、条件和监督自编码器、对抗式 SDF 估计和收益率曲线分解 —— 以及关于隐因子何时增加预测价值的跨数据集结果。

### [15. 因果机器学习](15_causal_estimation/)

Double Machine Learning 用于分离因子处理效应、贝叶斯结构时间序列用于事件影响评估，以及因果发现（PCMCI、NOTEARS、VAR-LiNGAM），应用于九个案例研究。

## 第四部分 — 策略实施（第16-20章）

从预测到可部署策略 —— 回测、组合构建、成本、风险和综合。

### [16. 策略模拟](16_strategy_simulation/)

回测即证伪：交易协议规范、向量化与事件驱动引擎、ETF 基线策略、核心指标报告、机制诊断，以及策略层面的过拟合控制（Deflated Sharpe Ratio、Rademacher Anti-Serum、White's Reality Check）。

### [17. 组合构建](17_portfolio_construction/)

从评分到组合：均值-方差优化及其陷阱、层次风险平价、凯利准则、共形仓位管理、深度组合配置，以及跨案例研究的受控配置器比较。

### [18. 交易成本](18_transaction_costs/)

成本分类、价差估计、市场冲击校准、执行算法（VWAP、TWAP、Almgren-Chriss 最优执行）、交易成本分析和实用防护措施 —— 盈亏平衡成本因资产类别差异很大。

### [19. 风险管理](19_risk_management/)

VaR/CVaR 尾部度量、回撤和路径风险控制、因子和行业分解、压力测试、自适应风险叠加层、深度对冲和熔断开关。叠加层效果因策略而异。

### [20. 策略综合](20_strategy_synthesis/)

九个实验揭示了将 ML 预测转化为策略的规律：IC-Sharpe 去相关、基本定律诊断、模型族级联、成本-生存分析、留出集失败模式，以及实践者决策框架。

## 第五部分 — 高级AI（第21-24章）

强化学习、大语言模型、知识图谱和金融自主智能体。

### [21. 强化学习用于执行与对冲](21_rl_execution_hedging/)

金融 MDP 建模、DQN/PPO/SAC 算法、最优执行、带库存管理的做市、PFHedge 深度对冲、用于策略恢复的逆强化学习，以及仿真到现实的差距。

### [22. 金融研究中的RAG](22_rag_financial_research/)

基于 SEC 文件的检索增强生成：数据摄取、领域特定嵌入、带重排序的混合检索、基于约束的提示、RAG 评估和故障诊断，以及向智能体工作流的过渡。

### [23. 知识图谱](23_knowledge_graphs/)

何时知识图谱值得其基础设施成本：从 SEC 文件构建知识图谱、用于多跳推理的 Graph RAG、用于 ML 的图特征（GNN 嵌入、中心性、社区检测）、金融网络和时间泄漏防护。

### [24. 自主智能体](24_autonomous_agents/)

智能体架构（ReAct、Tree of Thoughts、Reflexion）、记忆系统、工具契约、工程栈（LangGraph、Claude SDK）、有状态的股票研究智能体、带对抗辩论的多智能体预测，以及生产可靠性。

## 第六部分 — 生产（第25-26章）

策略上线 —— 交易系统及其持续运行的运营基础设施。

### [25. 实盘交易系统](25_live_trading/)

统一研究与生产的框架：Interactive Brokers 和 Alpaca 集成、托管平台（QuantConnect）、订单生命周期管理、流水线验证和运营就绪。

### [26. MLOps与治理](26_mlops_governance/)

ML 故障分类（流水线偏移 vs 性能衰退）、漂移检测、安全模型发布、熔断机制、特征存储、实验追踪，以及金融 ML 系统所需的 MLOps 基础设施。

## 结论

### [27. 系统性优势](27_systematic_edge/)

系统化哲学、量化职业路径、学习资源、研究前沿，以及如何构建自己的优势。与第1章首尾呼应：流程即优势。

---

## 快速开始

**初次接触？按顺序阅读以下三篇：**

1. **[本仓库是什么，不是什么](docs/what-this-is.md)** - 一条命令能复现什么、配置更改能带来什么、什么需要真实算力或授权数据、以及什么不做保证。五分钟阅读，在安装任何东西之前设定预期。
2. **[安装](docs/installation.md)** - Linux、Windows WSL2、macOS、Docker 和 GPU。
3. **[运行Notebook](docs/running-notebooks.md)** - 案例研究流水线、运行日志，以及如何在不影响已下载结果的情况下进行实验。

以下命令在你自己电脑的终端中输入，不是在 GitHub 中。不熟悉命令行？从 **[开始之前](docs/installation.md#before-you-begin)** 开始。

所有操作 **从仓库根目录执行**。使用 Docker 或本地 `uv` 环境克隆并设置：

```bash
git clone https://github.com/stefan-jansen/machine-learning-for-trading.git
cd machine-learning-for-trading
cp .env.example .env

docker compose pull ml4t # 方案A — Docker（推荐）
```

方案B 是本地 `uv` 环境，适用于 **macOS、Linux 或 WSL2 内**。使用 `uv` 自身的安装器而非 `pip` 安装 `uv`，因为 `pip` 在大多数当前系统上缺失或拒绝安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # 安装器自身的命令；立即将 uv 加入 PATH
uv sync
```

方案B 从源码编译多个依赖（包括 `scikit-learn`），因此需要 **C/C++ 编译器和 Python 头文件**：Ubuntu/Debian/WSL2 执行 `sudo apt install build-essential python3-dev`，macOS 执行 `xcode-select --install`。Docker 自带编译器，无需这些。

**macOS 用户：** **Apple Silicon** 选择方案B。这是每次发布前在真实硬件上走过的路径，只需 Xcode 命令行工具即可编译从源码构建的包。Docker 仅在以下情况值得占用磁盘空间：十二个 `ml4t-py312` Notebook（无 arm64 构建版本，已预执行）和第2章的容器化数据库基准测试。**Intel Mac** 选择方案A：PyTorch 不再发布 macOS x86_64 wheel，方案B 无法工作。

**Windows 用户：** 两种方案都在 WSL2 内运行，不在 PowerShell 中。从管理员 PowerShell 运行 `wsl --install -d Ubuntu`，重启，再运行一次（第一次通常只安装 WSL 运行时而没有发行版），然后在 Ubuntu 终端中按 Linux 说明操作。不支持在 Windows Python 中安装 —— `scikit-learn` 没有此 Python 版本的 Windows wheel，源码构建会失败。[安装指南](docs/installation.md) 提供完整的 WSL2 教程。

方案B 预留约 **16 GB**（11 GB 环境、4 GB 免费数据集、0.9 GB git 历史），数据下载约需12分钟。

详见 **[安装指南](docs/installation.md)** 了解平台特定设置和 GPU 说明。Intel Mac 仅限 Docker：PyTorch 不再发布 macOS x86_64 wheel，本地 `uv` 路径无法解析。

**下载数据。** 大多数 Notebook 需要数据集；从免费数据集开始（无需 API 密钥）：

```bash
uv run python data/download_all.py --free-only
```

Docker 用户在 Jupyter Lab 终端中运行（**File → New → Terminal**）`python data/download_all.py --free-only` —— Docker 路径没有宿主机 Python。

该命令获取七个数据集，约 **4 GB**，需约12分钟，其中绝大部分是公司特征面板，最早在第4章需要。如果只想用约 75 MB 开始，跳过它，在章节需要时再获取：

```bash
uv run python data/download_all.py --free-only --skip-firm-characteristics
```

**[数据指南](data/README.md)** 记录了每个数据集、API 密钥设置、加载器和存储层级。

**（可选）预计算结果。** 要在不重新训练的情况下探索九个已发布的第11-20章案例研究，下载其经过验证的注册表、预测、模型文件和回测产物：

```bash
uv run python scripts/download_artifacts.py
```

**运行Notebook。** Notebook 是成对的 [Jupytext](https://jupytext.readthedocs.io/) 文件（`.py` 源文件 + 生成的 `.ipynb`）。运行快速冒烟测试，或打开 Jupyter Lab：

```bash
uv run python 01_process_is_edge/factor_regimes.py                # 冒烟测试
ML4T_DATA_PATH="${ML4T_DATA_PATH:-$PWD/data}" uv run jupyter lab  # 本地：打开其打印的URL
docker compose up -d ml4t                                         # Docker：相同地址
```

`uv sync` 已安装 Jupyter Lab。从仓库根目录启动：`ML4T_DATA_PATH` 前缀为加载器提供绝对路径，因为 Jupyter 以章节文件夹作为工作目录运行每个 Notebook，否则会在该文件夹内搜索并报告数据集缺失。它会保留你已导出的值，并默认使用本仓库的 `data/`。

详见 **[运行Notebook](docs/running-notebooks.md)** 了解案例研究流水线、Papermill 参数和实验工作流。

### Docker 镜像

大多数 Notebook 在默认的 **ml4t** 镜像上运行；少数需要专用镜像，每个这样的 Notebook 在其前言中会注明。完整详情请参阅 **[Docker 环境指南](envs/README.md)**。

| 镜像         | 覆盖范围                                                     | 何时需要                |
|--------------|--------------------------------------------------------------|-------------------------|
| `ml4t`       | 全部27章 + 9个案例研究（CPU）                                | 所有内容的默认选择      |
| `ml4t-gpu`   | 同一 `ml4t` 镜像，使用 NVIDIA 运行时（`--profile gpu`）      | 深度学习章节            |
| `ml4t-py312` | Python 3.12，用于 signatory、esig、gensim、pfhedge、tfcausalimpact | 约10个Notebook    |
| `benchmark`  | 数据库客户端（TimescaleDB、ClickHouse、QuestDB、InfluxDB）   | 第2章存储基准测试       |
| `rapids`     | RAPIDS cuML + LightGBM CUDA（本地构建）                      | 一个第12章GPU基准测试   |

**寻找第二版？** 第二版在 `second-edition` 分支上完整且稳定 —— `git checkout second-edition`，一切都在书中描述的位置。

---

## 仓库结构

```text
machine-learning-for-trading/
├── 01_process_is_edge/ … 27_systematic_edge/   27章 — Jupytext .py + .ipynb，每章附README
├── case_studies/     九个数据集通过完整流水线（第6章 → 第20章）
├── data/             每个数据集的下载脚本和加载器                  → data/README.md
├── utils/            共享配置、路径、样式、建模和CV代码            → utils/README.md
├── scripts/          读者工具（安装检查、Notebook同步、产物下载）   → scripts/README.md
├── tests/            Papermill Notebook执行 + 单元测试，在CI中运行 → tests/README.md
├── envs/             每个镜像的Dockerfile                          → envs/README.md
├── docs/             仓库说明、安装和Notebook执行指南
├── docker-compose.yml    所有Docker服务
├── pyproject.toml · uv.lock    固定依赖（uv）
└── matplotlibrc      图表样式，从仓库根目录自动应用
```

---

## 贡献与反馈

发现错误、失效链接或有建议？在书籍出版前，早期反馈尤为宝贵。

- **问题**：[提交 GitHub Issue](https://github.com/stefan-jansen/machine-learning-for-trading/issues)
- **网站与联系方式**：[ml4trading.io](https://ml4trading.io)

---

## 许可证

代码：[MIT 许可证](LICENSE) · 书籍内容：© 2026 Stefan Jansen. 保留所有权利。

<p align="center">
  <a href="https://amzn.to/4eigy2F">获取本书</a> •
  <a href="https://ml4trading.io">ml4trading.io</a> •
  <a href="https://github.com/stefan-jansen/machine-learning-for-trading">GitHub</a>
</p>
