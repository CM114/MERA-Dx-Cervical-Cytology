# MERA-Dx-Cervical-Cytology 公开复现仓库设计

## 1. 目标

为 IEEE BIBM 2026 WS#4（Machine Learning for Biological and Medical Image Big Data）准备一个公开、可复现、适合导师审阅和论文投稿引用的 GitHub 仓库。

仓库名称确定为：

```text
MERA-Dx-Cervical-Cytology
```

仓库只覆盖当前论文的 MERA-Dx 宫颈细胞学方法、相关消融实验、对比实验、数据协议、结果汇总和绘图流程，不承载工作区内与本论文无关的历史试验、财务文件、临时文件或其他项目。

## 2. 公开范围

### 公开

- 论文相关训练、验证、测试、评估、审计和绘图代码；
- B0/C0/C1/C2/C3 与 MERA-Dx 主线实验配置；
- 论文中使用的对比方法适配器和统一评估脚本；
- 五折划分协议、随机种子、指标定义和实验流程；
- 不含本地路径和样本身份信息的表格/图源数据；
- 实验清单、结果来源映射、环境配置、复现说明和限制说明；
- 不含原始图像的示例 manifest、字段说明和数据准备接口。

### 不公开

- XUData、CRIC 或其他数据集的原始图像；
- 未确认授权的数据表、患者/切片/样本身份信息；
- 模型权重、训练缓存、完整逐图预测文件和大体积中间产物；
- 含本机或服务器绝对路径的日志、CSV、JSON 和可视化源文件；
- 未完成许可证审查的第三方仓库源码；
- 工作区中的临时目录、压缩包、PPT/财务文件和无关项目。

## 3. 目标仓库结构

```text
MERA-Dx-Cervical-Cytology/
├── README.md
├── CITATION.cff
├── LICENSE
├── .gitignore
├── environment.yml
├── configs/
│   ├── main/
│   ├── ablations/
│   └── baselines/
├── experiments/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── baselines/
│   ├── audits/
│   └── figures/
├── scripts/
├── tests/
├── data/
│   ├── README.md
│   ├── data_dictionary.md
│   └── example_manifest.csv
├── results/
│   ├── tables/
│   ├── figures_source_data/
│   └── experiment_summary.csv
├── docs/
│   ├── reproducibility.md
│   ├── experiment_inventory.md
│   ├── figure_table_mapping.md
│   ├── data_availability.md
│   └── limitations.md
└── tools/
    └── validate_public_release.py
```

实际整理时可以保留现有脚本的相对引用，优先通过包装脚本和配置入口实现路径无关化，避免大规模重命名导致论文结果无法复现。

## 4. 复现协议

公开仓库必须明确记录以下内容：

- 数据集版本和公开/受限获取方式；
- XUData TBS5 的五类标签定义；
- 固定五折划分、训练/验证规模和随机种子；
- 图像级、样本级或其他评估层级，避免误写为患者级或临床验证；
- 训练轮数、初始化方式、预处理和增强；
- Acc、Bal-Acc、Macro-Prec、Macro-Rec、Macro-Spec、Macro-F1、Macro-AUC 的定义；
- 每张表和每幅图对应的源文件、生成脚本和实验配置；
- 结果汇总中的均值、标准差、置信区间和统计单位；
- 当前实验限制：不能把开发阶段图像/crop 级结果表述为患者级、切片级或外部验证结果。

## 5. 结果发布策略

仓库只发布论文复核所需的紧凑结果：

1. 表格结果：脱敏的均值/标准差、折级汇总和必要的指标定义；
2. 图源数据：只保留绘图所需的数值列，移除图像路径、样本标识和服务器路径；
3. 实验清单：记录实验状态、代码入口、配置、结果文件和是否纳入论文；
4. 结果校验：发布前运行路径、敏感字段、权重扩展名和大文件扫描；
5. 可追溯性：每个公开结果映射到论文表格/图、脚本和配置。

## 6. GitHub 与引用

- 新建公开 GitHub 仓库 `MERA-Dx-Cervical-Cytology`；
- 公开前完成密钥、绝对路径、原始数据和大文件扫描；
- README 中分开写明 Code Availability 和 Data Availability；
- 代码许可证只在确认作者授权后加入，默认候选为 Apache-2.0；
- 不在仓库中臆造数据集许可证、DOI、访问限制或伦理审批信息；
- GitHub 仓库稳定后可进一步连接 Zenodo 获取归档 DOI；
- 为导师提供仓库链接和一段可直接转发的中文说明。

## 7. 验收标准

在创建 GitHub 仓库前，本地发布目录必须满足：

- `python tools/validate_public_release.py` 通过；
- 不包含 `.pt`、`.pth`、`.ckpt`、`.safetensors` 等模型权重；
- 不包含原始图像、患者信息、样本身份信息和绝对路径；
- README 能在没有本地工作区路径的情况下说明安装、数据准备和复现入口；
- 论文主要表格和图均有 source-data 与生成脚本映射；
- 代码、配置、结果摘要和文档之间的相对路径可用；
- Git 提交历史只包含本次公开仓库所需内容。

## 8. 当前需要保留的决策

- 仓库名：`MERA-Dx-Cervical-Cytology`；
- 可见性：Public；
- 目标会议：IEEE BIBM 2026 WS#4 / ML4BMI；
- 处理原则：代码尽量完整公开，数据和权重分离，结果可追溯；
- 投稿截止参考：2026-09-28；终稿截止参考：2026-11-02。

