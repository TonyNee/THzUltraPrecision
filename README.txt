================================================================================
  THzUltraPrecision — 太赫兹超高精度频率校准系统
================================================================================

项目简介
--------
本项目基于机器学习/深度学习方法，对太赫兹（THz）频率测量进行超高精度
校准。给定一个测量频率值 F_measured，模型预测对应的真实频率 F_true，
目标是将校准残差控制在亚 MHz 量级。

核心思想是采用残差学习范式：网络不直接预测绝对频率，而是学习一个微小
的修正量 ΔF，使得 F_pred = F_measured + ΔF 尽可能逼近 F_true。

================================================================================
  目录结构
================================================================================

THzUltraPrecision/
│
├── config.py              [核心] 全局配置管理器（Config类 + Utils工具类）
├── model.py               [核心] 神经网络模型定义与注册表
├── train.py               [核心] K-Fold 交叉验证训练流程
├── eval.py                [核心] 完整评估（分位数误差、残差PDF、分通道MAE）
├── eval_pred.py           [核心] 纯推理脚本（无标签，仅输出预测CSV）
├── eval_sfmm.py           [核心] 轻量评估 + 推理计时 + 直方图对比
├── export_onnx.py         [核心] PyTorch模型导出为ONNX格式
├── genlog.py              [核心] 实验日志汇总（收集config.yaml → CSV）
├── proc_input.py          [核心] 输入数据批量预处理
│
├── requirements.txt        Python环境依赖清单
│
├── input/                  输入数据目录（按日期组织）
│   ├── scale/111/          当前主用数据集（train.csv / eval.csv）
│   ├── 20250928/           历史数据集
│   ├── 20251216/
│   └── ...
│
├── output/                 训练/评估输出目录（按模型类型组织）
│   ├── resmlp/             各次实验的时间戳子目录（含config.yaml、模型、图表）
│   ├── bpnn/
│   └── demo/
│
├── experiment/             实验相关数据与结果存档
│
├── archive/                Git archive 分支快照（备份副本）
│
├── compare/                传统机器学习基线模型（与神经网络对比）
│   ├── 0-40ghz/            0~40 GHz 频段基线
│   ├── 100-1200ghz/        100~1200 GHz 频段基线
│   └── 200-740ghz/         200~740 GHz 频段基线
│   │
│   └── 每个频段下包含10种算法：
│       dt/     决策树回归 (Decision Tree)
│       knn/    K近邻回归 (K-Nearest Neighbors)
│       elastic/ 弹性网络 (ElasticNet)
│       gbdt/   梯度提升 (Gradient Boosting)
│       rf/     随机森林 (Random Forest)
│       ridge/  岭回归 (Ridge Regression)
│       linear/ 线性回归 (Linear Regression)
│       poly/   多项式回归 (Polynomial Regression)
│       lasso/  Lasso回归 (L1正则化)
│       svr/    支持向量回归 (SVR, RBF核)
│
├── backup/                 早期 ResMLP 实现备份（无K-Fold/YAML/注册表）
│   ├── config.py           简化版Config（硬编码路径）
│   ├── model_resmlp.py     HighPrecCalibrator 独立模型
│   ├── train_resmlp.py     简单train/eval划分训练
│   └── eval_resmlp.py      基础评估（MAE/STD/MSE + 残差图）
│
└── deprecated/             已废弃的实验代码
    ├── other/               LSTM 序列建模实验（已放弃）
    │   ├── lstm_model.py
    │   ├── train_lstm_calibration.py
    │   ├── eval_lstm_calibration.py
    │   └── bpnn_eval_multi.py
    │
    └── script/              BPNN 各版本脚本迭代记录
        ├── model.py         v1 BPNN（1-64-32-1, 无归一化）
        ├── model_paper.py   论文BPNN（500-50-10, tanh+purelin）
        ├── bpnn_train.py    v1训练（原始GHz, HuberLoss+AdamW）
        ├── bpnn_train_v2.py v2训练（Min-Max归一化, MSE+Adam）
        ├── bpnn_train_v3.py v3训练（Z-Score归一化, HuberLoss+AdamW）
        ├── bpnn_eval.py / bpnn_eval_v2.py / bpnn_eval_v3.py  对应评估
        └── train_highprec.py HighPrecNet残差网络训练

================================================================================
  核心工作流程
================================================================================

【流程1：训练 → 评估】

  第1步  配置参数
        ├── 编辑 config.py 顶部类属性：
        │   ├── MODEL_TYPE   = "resmlp"   （或 "bpnn" / "demo" / "linear"）
        │   ├── TRAIN_CSV     = "./input/scale/111/train.csv"
        │   ├── EVAL_CSV      = "./input/scale/111/eval.csv"
        │   ├── K_FOLDS       = 5
        │   └── （可选覆盖 BATCH_SIZE / LEARNING_RATE 等，留空则用模型默认值）

  第2步  训练模型
        │   python train.py
        │
        │   训练过程说明：
        │   ├── Config.init() 解析模型注册表 → 填充默认超参数 → 创建输出目录
        │   ├── 5折交叉验证 (K-Fold CV)，每折独立训练
        │   │   ├── 早停机制（val_loss 连续 PATIENCE 轮无改善则停止）
        │   │   └── 每折保存 best_model_fold_{k}.pth 和 loss_fold_{k}.png
        │   ├── 汇总CV结果 → 输出 cv_results.txt（含95%置信区间）
        │   ├── 全量数据重训练（epochs = avg(best_epochs)，防过拟合）
        │   └── 最终模型保存到 Config.MODEL_SAVE_PATH
        │
        │   输出目录示例：
        │   ./output/resmlp/20251218120000/
        │   ├── config.yaml                   ← 实验配置快照
        │   ├── resmlp_128-256-256-128-64.pth  ← 最终模型权重
        │   ├── best_model_cv.pth              ← CV最佳模型（备份）
        │   ├── best_model_fold_1.pth ...      ← 各折最佳模型
        │   ├── loss_fold_1.png ...            ← 各折loss曲线
        │   ├── loss_full_train.png            ← 全量训练loss曲线
        │   └── cv_results.txt                 ← CV统计结果
        │

  第3步  评估模型
        │   python eval.py --mdir ./output/resmlp/20251218120000/
        │
        │   评估流程：
        │   ├── 从 config.yaml 恢复配置 → 加载模型权重
        │   ├── 在评估集上推理
        │   ├── 计算指标：MAE / MSE / RMSE / R²
        │   │             + E1σ/E2σ/E3σ（68%/95%/99.7%分位数误差）
        │   │             + E80/E90/E95/E99
        │   ├── 更新 config.yaml（写入evaluation节）
        │   ├── 保存 data_predicted_{model}.csv（含预测值和残差）
        │   ├── 绘制 residual_plot_{model}.png（测量误差vs预测误差）
        │   ├── 绘制 residual_pdf_{model}.png（残差概率密度+高斯拟合）
        │   └── 打印分通道（低频/高频, 以747GHz为界）MAE
        │

【流程2：纯推理（无标签）】
        python eval_pred.py --mdir ./output/resmlp/20251218120000/
        → 读取 Eval CSV（只需第一列测量频率）→ 输出 pred.csv

【流程3：轻量评估 + 推理计时】
        python eval_sfmm.py --mdir ./output/resmlp/20251218120000/
        → 额外的推理时间统计 + 测量值/预测值直方图对比

【流程4：导出ONNX部署】
        python export_onnx.py --mdir ./output/resmlp/20251218120000/
        → 导出 resmlp.onnx 文件，可用于C++/嵌入式端推理

【流程5：实验日志汇总】
        python genlog.py
        → 遍历 ./output/resmlp/ 下所有子目录，收集 config.yaml →
          扁平化 → 汇总为 log.csv（Excel/Pandas可直接打开对比）

【流程6：输入数据预处理】
        修改 proc_input.py 中的 input_dir / output_dir 路径，
        编辑处理逻辑（当前为：第一列减去0.2 GHz偏移），运行：
        python proc_input.py

================================================================================
  compare 基线模型运行
================================================================================

compare/ 目录包含30个独立的基线实验脚本（3个频段 × 10种算法）。
每个脚本是自包含的，不依赖根目录的 Config/model 模块。

运行方式（以 Decision Tree 为例）：
  cd compare/0-40ghz/dt/
  python main.py

输出：
  - 最佳超参数和CV MSE（打印到终端）
  - 评估集MAE/MSE/RMSE/MaxAE
  - 综合分析图（散点+预测vs实际+残差）
  - 频率残差对比图（MHz）

注意：compare 脚本的数据路径为相对路径 ./input/20251216/train.csv，
      需要确保从脚本所在目录或有对应数据的目录运行。

================================================================================
  CSV 数据格式
================================================================================

训练/评估数据为两列CSV，带表头行：

  Fexperiment_GHz,Fstandard_GHz
  3.626577479,3.511448036
  3.684142201,3.597795118
  ...

  第1列 (Fexperiment_GHz): 仪器测量的频率值（GHz）
  第2列 (Fstandard_GHz):   标准参考频率值（GHz，即"真实值"）

纯推理模式（eval_pred.py）的输入CSV可以只有第一列，无表头。

================================================================================
  可用模型一览
================================================================================

模型通过 config.py 的 MODEL_TYPE 选择，所有模型定义在 model.py 中：

  MODEL_TYPE = "resmlp"  → ResMLP（推荐）
    残差MLP: 1→128→256→256→128→64→1, SiLU激活
    残差连接: output = input + net(input)
    默认: batch=32, lr=2e-5, epochs=10000, patience=500, MSE+Adam+CosineAnnealing

  MODEL_TYPE = "bpnn"    → BpnnPaper（论文复现）
    500-50-10 隐藏层, tanh(第1层) + purelin(后续层)
    默认: batch=64, lr=0.035, epochs=100000, patience=5000, Huber+Rprop

  MODEL_TYPE = "demo"    → BPNN（简化演示）
    1→64→128→64→1, ReLU激活
    默认: batch=64, lr=0.035, epochs=100000, patience=2000, Huber+Rprop

  MODEL_TYPE = "linear"  → LinearRegression（线性基线）
    单层 y = wx + b
    默认: batch=64, lr=0.01, epochs=100000, patience=20000, MSE+Adam

自定义模型：继承 nn.Module，设置 DEFAULT_CONFIG，用 @register_model 注册即可。

================================================================================
  训练超参数继承机制
================================================================================

Config 类中的超参数遵循三层优先级（由高到低）：

  1. 用户在 config.py 中显式设置的值（如 BATCH_SIZE = 128）
  2. 若未设置（None），则从模型类的 DEFAULT_CONFIG 字典中读取
  3. 不允许缺失——DEFAULT_CONFIG 必须覆盖所有必要字段

示例：ModelType="resmlp" 时，若未显式设 LEARNING_RATE，
      则自动从 ResMLP.DEFAULT_CONFIG 中读取 learning_rate=2e-5。

================================================================================
  Python 环境依赖
================================================================================

核心依赖（详见 requirements.txt）：

  pip install -r requirements.txt

主要包：
  torch >= 2.8.0        深度学习框架
  numpy                  数值计算
  pandas                 数据处理（CSV读写）
  scikit-learn           传统ML基线（compare/）+ KFold划分
  scipy                  统计分布（compare/中的随机搜索）
  matplotlib             可视化绘图
  PyYAML                 配置文件读写
  onnx                   ONNX模型导出（export_onnx.py）

建议使用虚拟环境或 conda 管理依赖。

================================================================================
  常见问题与注意事项
================================================================================

Q1: 如何更换训练数据？
A1: 修改 config.py 中的 TRAIN_CSV 和 EVAL_CSV 路径，然后运行 train.py。

Q2: 如何调整 K-Fold 折数？
A2: 修改 config.py 中的 K_FOLDS 和 CV_SEED。

Q3: 评估时报 "Config yaml not found" 怎么办？
A3: 确认 --mdir 参数指向的目录中确实有训练生成的 config.yaml。

Q4: backup/ 和 deprecated/ 的代码还有用吗？
A4: 仅作历史参考，不再维护。核心流程请使用根目录的 config/model/train/eval。

Q5: 想用 compare/ 的基线模型跑新数据怎么办？
A5: 修改对应 compare/{频段}/{算法}/main.py 中的 CSV 路径即可。

Q6: 为什么残差单位有时是 GHz 有时是 MHz？
A6: 内部计算统一使用 GHz（匹配CSV原始数据），显示和评估指标使用 MHz
    （1 GHz = 1000 MHz），残差乘以1000即可转换。

Q7: 747 GHz 是什么意思？
A7: 太赫兹系统的通道分界频率。低频段(<747GHz)和高频段(>747GHz)
    使用不同的硬件通道，分别评估有助于诊断各通道的校准效果。

Q8: ResMLP 的残差连接为什么重要？
A8: 频率值本身很大（数百GHz），但修正量很小（MHz级别）。
    如果让网络直接输出绝对频率，梯度信号会被大数值淹没。
    残差学习让网络只需输出一个微小修正量，大大降低了优化难度。

================================================================================
  分支说明
================================================================================

  main          — 主分支（精简后的稳定版本）
  grad/thesis   — 学位论文分支（包含完整实验代码和历史迭代，当前分支）
  archive       — grad/thesis 的备份快照（注释添加前的原始状态）

================================================================================
  作者与许可
================================================================================

作者: TonyNee
项目: THzUltraPrecision — 太赫兹超高精度频率校准
用途: 学术研究 / 学位论文
