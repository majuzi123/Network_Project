# 网络异常检测项目

## 项目概述

这是一个基于**图形特征**和**支持向量机（SVM）**的IP网络流量异常检测系统。该系统通过为每个源IP地址构建一个**Graphlet**（小图）作为其通信特征，然后使用SVM将主机分类为正常或恶意。

## 核心概念

### Graphlet（图形特征）
- 每个源IP地址的通信特征表示为一个有向图
- 节点代表通信中的各个要素（IP地址、协议、端口等）
- 边表示这些要素之间的关系
- 编码了该主机的通信模式

### 随机游走核
- 通过计算两个图中相同随机游走路径的数量来衡量相似性
- 长度为k的随机游走是图中k+1个节点的序列
- 能有效捕捉图形的结构特征
- 支持核技巧优化，避免显式特征映射

### 支持向量机（SVM）
- 在高维特征空间中训练的二分类器
- 将主机分为"正常"和"恶意"两类
- 支持多种核函数

## 项目结构

```
NetworkProject/
├── main.py                          # 主程序入口
├── requirements.txt                 # 依赖包列表
├── Network_Anomaly_Detection.ipynb   # Jupyter Notebook（完整项目）
├── data/
│   ├── annotated-trace.csv          # 标注的网络流量数据
│   └── not-annotated-trace.csv      # 未标注的网络流量数据
├── src/
│   ├── __init__.py
│   ├── anomaly_detector.py          # 异常检测模块
│   ├── graphlet_builder.py          # Graphlet构建模块
│   └── random_walk_kernel.py        # 随机游走核模块
├── output/                          # 输出结果目录
└── README.md                        # 项目说明文档
```

## 安装依赖

### 方法1：使用requirements.txt

```bash
pip install -r requirements.txt
```

### 方法2：手动安装

```bash
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install networkx>=2.6.0
pip install scikit-learn>=0.24.0
pip install matplotlib>=3.4.0
```

## 使用方法

### 1. 运行Jupyter Notebook

最全面的方式是运行Jupyter Notebook，它包含所有步骤和可视化：

```bash
jupyter notebook Network_Anomaly_Detection.ipynb
```

### 2. 数据格式

#### 标注数据（annotated-trace.csv）

CSV格式，每行一条记录：
```
srcIP,dstIP,protocol,sPort,dPort,label
742,281,17,53,22,normal
241,591,6,53,53,normal
537,732,17,23,22,malicious
...
```

字段说明：
- `srcIP`: 源IP地址（整数）
- `dstIP`: 目标IP地址（整数）
- `protocol`: 协议类型（1=ICMP, 6=TCP, 17=UDP）
- `sPort`: 源端口
- `dPort`: 目标端口
- `label`: 标签（normal 或 malicious）

#### 未标注数据（not-annotated-trace.csv）

CSV格式，无标签列：
```
srcIP,dstIP,protocol,sPort,dPort
818,372,6,80,53
16,822,1,25,23
472,327,6,80,53
...
```

## 项目步骤

### 第1部分：安装和导入库
- 安装必要的Python包
- 导入scikit-learn、NetworkX、Pandas等库

### 第2部分：数据读取和解析
- 从CSV文件加载网络流量数据
- 解析和组织流量记录

### 第3部分：构建Graphlet
- 为每个源IP地址构建图形表示
- 节点代表通信要素，边代表关系

### 第4部分：随机游走核实现
- 生成长度为4的随机游走
- 计算两个图形之间的相似性

### 第5部分：特征映射
- 将随机游走转换为特征向量
- 统一特征维度

### 第6部分：SVM分类（直接映射）
- 将特征向量输入SVM
- 训练和评估模型

### 第7部分：SVM分类（核技巧）
- 计算核矩阵
- 使用预计算核函数训练SVM
- 对比两种方法的性能

### 第8部分：异常检测
- 对未标注数据进行预测
- 识别恶意主机

### 第9部分：分析异常流量
- 分析检测到的异常主机的特征
- 推测可能的攻击类型

### 第10部分：假正例和假负例分析
- 分析模型错误情况
- 讨论改进策略

### 第11部分：无振荡随机游走
- 实现避免振荡的随机游走算法
- 重新训练模型并对比结果

## 性能指标

模型使用以下指标进行评估：

- **准确率 (Accuracy)**: 正确分类的样本比例
- **精确率 (Precision)**: 预测为恶意的样本中实际恶意的比例
- **召回率 (Recall)**: 实际恶意的样本中被正确识别的比例
- **F1-分数**: 精确率和召回率的调和平均数
- **混淆矩阵**: 真正例、真负例、假正例、假负例的分布

## 关键发现

1. **Graphlet表示有效**：能够捕捉主机的通信特征
2. **随机游走核强大**：是一种有效的图形相似性度量
3. **SVM性能良好**：在该问题上表现出优秀的分类能力
4. **无振荡改进**：改进的随机游走方法提供了更好的特征表示
5. **检测可行**：系统能够成功识别网络中的异常活动

## 攻击类型推测

系统可以识别以下类型的异常流量：

- **DDoS攻击**：大量UDP流量或多个源到同一目标的流量
- **端口扫描**：多个不同目标端口的流量
- **Web应用攻击**：针对端口80、443的流量
- **系统服务攻击**：针对低端口（<1024）的流量
- **DNS放大攻击**：特定端口（如53）的异常流量

## 改进建议

1. **特征工程**：加入时间序列特征、连接统计等
2. **模型优化**：尝试其他核函数、超参数调优
3. **检测策略**：多层次检测、异常度评分
4. **实际应用**：增量学习、事件响应集成

## 参考论文

Karagiannis, T., Papagiannaki, K., Taft, N., and Faloutsos, M. (2007, April). 
Profiling the end host. In International Conference on Passive and Active Network Measurement (PAM) (pp. 186-196). Springer, Berlin, Heidelberg.

## 作者和许可证

本项目为教育目的实现。

## 联系方式

如有问题或建议，欢迎反馈。

---

**最后更新**: 2026年1月7日
