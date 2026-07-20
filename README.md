# CSI 数据分析与动作识别

本目录包含 CSI 数据的可视化标注、预处理、数据集划分和分类模型训练代码。当前实现支持两类原始数据：

- Nexmon CSI 采集程序导出的 CSV（`amplitude` 字段为 JSON 数组）
- Intel 5300 CSI Tool 生成的 `.dat` 文件

主要代码位于 [`CSI_Data_Analysis`](./CSI_Data_Analysis/) 目录。

## 目录结构

```text
csi_analysis/
├─ README.md
└─ CSI_Data_Analysis/
   ├─ advanced_annotator.py   # CSV 批量可视化标注与滑窗特征提取（推荐）
   ├─ 数据标注.py             # Intel 5300 .dat 文件自由标注工具
   ├─ generate_dataset.py     # 按 CSV 所在目录自动生成三分类数据集
   ├─ split_dataset.py        # 将手动标注数据按 70/15/15 划分
   ├─ train_pytorch.py        # PyTorch 1D CNN 训练
   ├─ train_sklearn.py        # PCA + 随机森林基线
   ├─ DataNpy20260716/        # 已生成的数据、模型和训练曲线
   └─ DataNpy20260720/        # 已生成的数据、模型和训练曲线
```

## 环境准备

建议使用 Python 3.10 及以上版本：

```powershell
cd E:\develop\masm\csi_analysis\CSI_Data_Analysis
pip install numpy pandas scipy matplotlib scikit-learn torch csiread
```

`tkinter` 用于标注工具的图形界面，Windows 官方 Python 通常已自带。

## CSV 标注流程（推荐）

1. 启动标注工具：

   ```powershell
   python advanced_annotator.py
   ```

2. 选择多个 CSV 文件或包含 CSV 的文件夹。
3. 调整窗口大小、步长和搜索范围，也可以在图中点击指定动作起点。
4. 选择标签并将当前窗口加入数据集。
5. 导出 `X_custom.npy` 和 `y_custom.npy`。
6. 修改 `split_dataset.py` 顶部的 `data_dir`，然后执行：

   ```powershell
   python split_dataset.py
   ```

划分比例为训练集 70%、验证集 15%、测试集 15%。原始数据会被改名为 `X_all.npy` 和 `y_all.npy`；训练集在脚本中仍保存为 `X_custom.npy` 和 `y_custom.npy`。

CSV 至少需要以下字段：

- `local_timestamp_us`：本地微秒时间戳，用于估算采样率
- `amplitude`：JSON 数组格式的 CSI 幅度数据

`advanced_annotator.py` 的标签定义为：

| 标签 | 含义 |
| ---: | --- |
| 0 | Normal（静止） |
| 1 | Lying Down（躺下） |
| 2 | Falling（跌倒） |
| 3 | Walking（行走） |

勾选“PCA 降维 + STFT 特征提取”后，每个滑窗先经过中值滤波和 0.5 Hz 高通滤波，再提取第一主成分及 STFT 特征。

## CSV 自动生成三分类数据集

`generate_dataset.py` 会根据 CSV 所在文件夹的名称自动分配标签：

| 文件夹名称规则 | 标签 | 含义 |
| --- | ---: | --- |
| 包含 `fall` | 2 | Falling（跌倒） |
| 包含 `lie` | 1 | Lying Down（躺下） |
| 其他名称 | 0 | Walking/Normal（行走/日常） |

运行前修改文件顶部的 `SHUJU_DIR` 和 `OUTPUT_DIR`，再执行：

```powershell
python generate_dataset.py
```

脚本会先按原始文件进行分层划分，再提取滑动窗口，避免同一个 CSV 的相邻窗口同时进入训练集和测试集。默认窗口长度为 250、步长为 10，输出 `X_train.npy`、`X_val.npy`、`X_test.npy` 及对应标签。

## Intel 5300 `.dat` 标注流程

```powershell
python 数据标注.py
```

在界面中加载 `.dat` 文件，拖动选择动作片段，选择标签后导出 `.npy` 数据集。该工具提供 Raw、Highpass、PCA 和 STFT 四种视图，标签为走、坐下、站起、跌倒和其他。

## 模型训练

### PyTorch 1D CNN

修改 `train_pytorch.py` 顶部的 `DATA_DIR` 后运行：

```powershell
python train_pytorch.py
```

训练程序会自动选择 CUDA 或 CPU，并在数据目录中保存：

- `best_csi_model.pth`：验证集准确率最高的模型
- `pytorch_loss_curve.png`：训练集与验证集损失曲线

### PCA + 随机森林基线

修改 `train_sklearn.py` 顶部的 `DATA_DIR` 后运行：

```powershell
python train_sklearn.py
```

程序会输出验证集、测试集指标，并保存 `sklearn_confusion_matrix.png`。

## 运行前注意

- 多个脚本仍使用硬编码的 Windows 数据路径，运行前必须修改顶部的 `DATA_DIR`、`SHUJU_DIR` 或 `OUTPUT_DIR`。
- `train_pytorch.py` 当前固定为 3 个类别、6 个输入通道，并读取 `X_custom.npy`；若数据的标签数、特征通道数或文件名不同，需要同步修改模型参数和加载路径。
- `DataNpy20260716` 的特征形状为 `(样本数, 25, 6)`，而 `DataNpy20260720` 为 `(样本数, 25, 4)`，两者不能直接混合训练。
- `generate_dataset.py` 生成的是原始滑窗幅度特征，默认形状约为 `(样本数, 250, 192)`；它适用于随机森林脚本，但不能不经调整直接送入当前 1D CNN。
- 执行 `split_dataset.py` 前建议备份数据；脚本会重命名原始的 `X_custom.npy` 和 `y_custom.npy`。
