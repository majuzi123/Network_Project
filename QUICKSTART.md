# 快速开始指南

## 目录

1. [项目目标](#项目目标)
2. [快速安装](#快速安装)
3. [运行项目](#运行项目)
4. [关键步骤](#关键步骤)
5. [预期结果](#预期结果)
6. [常见问题](#常见问题)

## 项目目标

开发一个**网络异常检测系统**，能够：
- 为每个IP主机构建图形特征（Graphlet）
- 使用随机游走核提取特征
- 训练SVM分类器识别恶意主机
- 在未标注流量中检测异常

## 快速安装

### 1. 安装Python 3.7+
```bash
# macOS
brew install python3

# 或从 python.org 下载
```

### 2. 安装依赖
```bash
cd NetworkProject
pip install -r requirements.txt
```

### 3. （可选）使用虚拟环境
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

## 运行项目

### 方法1：使用启动脚本（推荐）
```bash
bash run.sh
```

### 方法2：直接启动Jupyter
```bash
jupyter notebook Network_Anomaly_Detection.ipynb
```

### 方法3：运行Python脚本
```bash
python3 main.py
```

## 关键步骤

### 第1步：数据准备
确保在 `data/` 目录中有：
- `annotated-trace.csv` - 标注的网络流量
- `not-annotated-trace.csv` - 未标注的网络流量

### 第2步：构建Graphlet
```python
from src.graphlet_builder import GraphletConstructor

constructor = GraphletConstructor()
constructor.build_all_graphlets(labeled_hosts)
```

### 第3步：提取特征
```python
from src.random_walk_kernel import RandomWalkKernel

walk_kernel = RandomWalkKernel(walk_length=4)
walks = walk_kernel.generate_all_walks(graph)
features = walk_kernel.walks_to_features(walks)
```

### 第4步：训练SVM
```python
from sklearn.svm import SVC

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
```

### 第5步：异常检测
```python
# 对未标注数据预测
anomalies = svm.predict(X_unlabeled)
confidence = svm.decision_function(X_unlabeled)
```

## 预期结果

成功运行后，你将获得：

### 1. 数据统计
- 标注数据中的正常和恶意流量比例
- 涉及的主机数量和流量数

### 2. 模型性能指标
```
准确率: 0.8500+
精确率: 0.80+
召回率: 0.85+
F1-分数: 0.82+
```

### 3. 可视化结果
- Graphlet网络图
- 随机游走示例
- 混淆矩阵
- ROC曲线
- 方法对比图表

### 4. 异常检测结果
- 检测到的恶意主机列表
- 异常流量特征分析
- 可能的攻击类型推测

### 5. 比较分析
- 直接映射 vs 核技巧方法
- 标准游走 vs 无振荡游走
- 性能和计算时间对比

## 常见问题

### Q: 如何解决 "ModuleNotFoundError: No module named 'networkx'"？
**A:** 运行 `pip install networkx`

### Q: 数据文件找不到？
**A:** 确保数据文件在 `data/` 目录中，文件名完全匹配

### Q: 特征维度太高导致内存不足？
**A:** 
- 减少每个节点生成的随机游走数
- 使用哈希技巧降低维度
- 使用核技巧方法

### Q: SVM训练速度很慢？
**A:**
- 使用核技巧方法（避免显式特征映射）
- 减少训练样本数
- 调整SVM参数（如C值）

### Q: 检测结果中假正例太多？
**A:**
- 增加SVM的C参数
- 降低分类阈值
- 调整正负样本权重

### Q: 如何改进检测性能？
**A:**
- 添加更多特征（时间、统计信息等）
- 使用更长的随机游走
- 尝试其他核函数（RBF、多项式）
- 进行超参数优化

## 项目结构说明

```
NetworkProject/
├── Network_Anomaly_Detection.ipynb  ← 主要的Jupyter Notebook
│                                      包含全部11个步骤和分析
│
├── src/                             ← Python模块
│   ├── graphlet_builder.py          构建图形特征
│   ├── random_walk_kernel.py        随机游走核
│   └── anomaly_detector.py          异常检测
│
├── data/                            ← 数据文件
│   ├── annotated-trace.csv          标注数据
│   └── not-annotated-trace.csv      未标注数据
│
├── output/                          ← 结果输出
│
├── main.py                          简单的启动脚本
├── requirements.txt                 依赖列表
├── README.md                        详细说明文档
├── QUICKSTART.md                    本文件
└── run.sh                          启动脚本
```

## 进阶用法

### 自定义随机游走长度
```python
kernel = RandomWalkKernel(walk_length=6)
```

### 使用不同的核函数
```python
svm = SVC(kernel='rbf', gamma='scale')
svm = SVC(kernel='poly', degree=3)
```

### 调整SVM正则化参数
```python
svm = SVC(kernel='linear', C=10.0)  # C越大，越严格
```

### 添加样本权重处理不平衡数据
```python
svm = SVC(kernel='linear', class_weight='balanced')
```

## 下一步

1. **阅读论文**：Karagiannis et al. (2007) - Profiling the end host
2. **扩展项目**：
   - 添加时间维度特征
   - 实现增量学习
   - 集成多个检测方法
3. **部署应用**：
   - 实时流量监控
   - 警告和响应系统
   - Web仪表板

## 参考资源

- [NetworkX文档](https://networkx.github.io/)
- [scikit-learn SVM文档](https://scikit-learn.org/stable/modules/svm.html)
- [Jupyter Notebook文档](https://jupyter.org/)

---

**祝你项目顺利！** 如有问题，请查看完整的 README.md 文件。
