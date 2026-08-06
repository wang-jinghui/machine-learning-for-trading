# 第1章：流程即优势

本章确立核心论点：在交易中，持久的业绩表现更多取决于维持一套能够应对市场变化、噪声信号和现实摩擦的纪律性研究流程，而非选择某个复杂模型。它为读者提供一套关于市场变化的实用词汇，展示为何近期的冲击暴露了脆弱的假设，并将机器学习交易重新定义为一个适应问题，而非模型选择竞赛。

## 学习目标

* 区分结构性断裂、机制、数据漂移、概念漂移和在线检测，并解释为何静态交易模型在不断变化的市场中会退化
* 解释 ML4T 工作流程作为研究到生产系统的构成，包括其数据基础设施基础、范围界定不变量、迭代研究模块，以及从实盘交易返回研究的反馈回路
* 定义探索与确认之间的证据边界，并解释试验日志、密封留出集和选择感知评估如何维护研究完整性
* 描述因果推断和生成式AI如何在纪律性交易工作流程中定位，包括它们提供的主要好处以及引入的新失败模式
* 应用机制思维、可实施性检查和监控逻辑来诊断策略脆弱性，并在独立和机构环境中调整工作流程纪律

## 章节

### 1.1 为何流程纪律至关重要

本节确立本章的核心论点：在交易中，持久的业绩表现更多取决于维持一套能够应对市场变化、噪声信号和现实摩擦的纪律性研究流程，而非选择某个复杂模型。它为读者提供一套关于市场变化的实用词汇，展示为何近期的冲击暴露了脆弱的假设，并将机器学习交易重新定义为一个适应问题，而非模型选择竞赛。

### 1.2 介绍 ML4T 工作流程

本节呈现全书的核心框架：一个建立在时间点正确数据基础设施之上的研究到生产工作流，包含明确的范围界定规则、迭代式特征与模型开发、逼真的策略设计、部署纪律和持续监控。对读者的核心价值在于，它将交易研究转化为一个具有可审计产物、清晰交接和明确探索-确认边界的管理生命周期。

### 1.3 工作流程中的因果推断与生成式AI

本节将两个现代方法族置于工作流程之中，而非将其视为独立趋势。因果推断被定位为锐化机制、假设和诊断的工具；生成式AI被定位为扩展研究和非结构化数据处理的工具，同时也带来新的风险，如泄漏、幻觉和工作流膨胀。读者应当关注的是，本节清楚地表明：新工具提升了纪律的价值，而非取代纪律。

### 1.4 市场机制：变化是永恒的

本节将非平稳性转化为可操作的内容。它展示机制概念如何支持解释、稳健性检查和实时监控，同时强调机制主要是一种风险视角，而非可靠的择时信号。因子和宏观示例使这一概念具体化：当机制方法有助于识别不利环境并将其与预定义的风险行动关联时，它们才真正有用。

- [`factor_regimes`](factor_regimes.ipynb) — 演示使用高斯混合模型（GMM）对 AQR 百年因子溢价数据集中的因子收益进行无监督学习，实现市场机制检测。
- [`macro_regimes`](macro_regimes.ipynb) — 演示使用 FRED 的宏观经济指标进行无监督学习以实现市场机制检测，并通过 S&P 500 波动率和回撤进行验证。

### 1.5 现实世界：独立 vs. 机构

本节将工作流程转化为实际运营场景。它解释了机构如何从内置的摩擦和审查中受益，而独立研究者则必须通过文档、检查点和明确的停止标准来创建自己的治理体系。实际收益非常显著：它帮助读者看到独立从业者的脆弱之处、仍可竞争的领域，以及可复用基础设施如何随时间推移持续提升研究质量。

## 运行 Notebook

```bash
# 从仓库根目录执行
uv run python 01_process_is_edge/<notebook>.py

# 测试模式（通过 Papermill 使用缩减数据）
uv run pytest tests/test_notebooks.py -v -k "01_process_is_edge"
```

## 参考文献

- **Andrew Ang and Geert Bekaert** (2002). [International Asset Allocation With Regime Shifts](https://doi.org/10.1093/rfs/15.4.1137). *Review of Financial Studies*.
- **Robert D. Arnott et al.** (2018). [A Backtesting Protocol in the Era of Machine Learning](https://doi.org/10.2139/ssrn.3275654).
- **Darrell Duffie** (2020). [Still the World's Safe Haven? Redesigning the U.S. Treasury Market After the COVID-19 Crisis](https://www.brookings.edu/wp-content/uploads/2020/05/WP62_Duffie_v2.pdf).
- **David Easley et al.** (2012). [The Volume Clock: Insights into the High Frequency Paradigm](https://doi.org/10.2139/ssrn.2034858).
- **Frank J. Fabozzi et al.** (2024). [Paradigm Shift: Embracing Holism in Causal Modeling for Investment Applications](https://doi.org/10.3905/jpm.2024.51.1.159). *The Journal of Portfolio Management*.
- **Frank J. Fabozzi and Caleb C. Stenholm** (2025). [Strategic Discipline: How Asset Management Mirrors Military Operations](https://doi.org/10.3905/jpm.2025.1.769). *The Journal of Portfolio Management*.
- **Ziang Fang and Jason Moore** (2025). What AI Can (and Can't Yet) Do for Alpha.
- **Stefano Giglio et al.** (2022). [Factor Models, Machine Learning, and Asset Pricing](https://doi.org/10.1146/annurev-financial-101521-104735). *Annual Review of Financial Economics*.
- **Campbell R. Harvey et al.** (2016). [...and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059). *Review of Financial Studies*.
- **Blanka Horvath et al.** (2021). [Clustering Market Regimes Using the Wasserstein Distance](https://doi.org/10.2139/ssrn.3947905).
- **Antti Ilmanen et al.** (2021). [How Do Factor Premia Vary Over Time? A Century of Evidence](https://doi.org/10.2139/ssrn.3400998).
- **Justina Lee** (2025). [Man Group Says Agentic AI Is Now Devising Quant Trading Signals](https://www.bloomberg.com/news/articles/2025-07-10/man-group-says-agentic-ai-is-now-devising-quant-trading-signals). *Bloomberg.com*.
- **Andrew W. Lo** (2004). [The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective](https://papers.ssrn.com/abstract=602222).
- **Martin Luk** (2023). [Generative AI: Overview, Economic Impact, and Applications in Asset Management](https://doi.org/10.2139/ssrn.4574814).
- **Judea Pearl** (2019). [The seven tools of causal inference, with reflections on machine learning](https://doi.org/10.1145/3241036). *Communications of the ACM*.
- **Marcos López de Prado** (2018). The 10 Reasons Most Machine Learning Funds Fail. *The Journal of Portfolio Management*.
- **Marcos Lopez de Prado et al.** (2024). [The Case for Causal Factor Investing](https://doi.org/10.2139/ssrn.4774522).
- **Marcos López de Prado and Vincent Zoonekynd** (2025). [Correcting the Factor Mirage: A Research Protocol for Causal Factor Investing](https://doi.org/10.3905/jpm.2025.1.794). *The Journal of Portfolio Management*.
- **James Ryseff et al.** (2024). [The Root Causes of Failure for Artificial Intelligence Projects and How They Can Succeed: Avoiding the Anti-Patterns of AI](https://www.rand.org/pubs/research_reports/RRA2680-1.html).
- **Bernhard Schölkopf et al.** (2021). [Towards Causal Representation Learning](https://doi.org/10.48550/arXiv.2102.11107).
- **Stefan Studer et al.** (2021). [Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology](https://doi.org/10.3390/make3020020). *Machine Learning and Knowledge Extraction*.
- **A. Sinem Uysal and John M. Mulvey** (2021). [A Machine Learning Approach in Regime-Switching Risk Parity Portfolios](https://doi.org/10.3905/jfds.2021.1.057). *The Journal of Financial Data Science*.
