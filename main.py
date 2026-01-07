#!/usr/bin/env python3
"""
网络异常检测数据诊断脚本
用于分析网络流量数据，特别关注anomaly标签的分布情况
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_network_flow(flow_string):
    """
    解析网络流量字符串
    格式: srcIP,dstIP,protocol,sPort,dPort[,label]
    支持标签: normal, anomaly 或 0,1
    """
    parts = flow_string.strip().split(',')
    
    if len(parts) < 5:
        return None
    
    try:
        src_ip = int(parts[0])
        dst_ip = int(parts[1])
        protocol = int(parts[2])
        src_port = int(parts[3])
        dst_port = int(parts[4])
        
        # 处理标签（可选）
        label = None
        if len(parts) >= 6:
            label_str = parts[5].strip().lower()
            
            # 处理文本标签
            if label_str == 'normal':
                label = 0
            elif label_str == 'anomaly' or label_str == 'malicious':
                label = 1
            else:
                # 尝试转换为数字
                try:
                    label = int(label_str)
                    # 确保标签是0或1
                    if label not in [0, 1]:
                        print(f"警告: 标签值 {label} 不是0或1")
                        label = None
                except:
                    label = None
        
        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'src_port': src_port,
            'dst_port': dst_port,
            'label': label
        }
    except Exception as e:
        print(f"解析错误: {e}, 行: {flow_string}")
        return None

def load_and_analyze_data(filepath, dataset_name="数据集"):
    """加载并分析网络流量数据"""
    print(f"\n{'='*80}")
    print(f"分析 {dataset_name}: {filepath}")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"错误: 文件不存在: {filepath}")
        return [], {}
    
    flows = []
    host_flows = defaultdict(list)
    label_stats = defaultdict(int)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    
    for line_num, line in enumerate(lines):
        flow = parse_network_flow(line)
        if flow:
            flows.append(flow)
            host_flows[flow['src_ip']].append(flow)
            
            # 统计标签
            if flow['label'] is not None:
                if flow['label'] == 0:
                    label_stats['normal'] += 1
                elif flow['label'] == 1:
                    label_stats['anomaly'] += 1
            else:
                label_stats['无标签'] += 1
    
    print(f"成功解析的流量记录: {len(flows)}")
    print(f"涉及的主机数: {len(host_flows)}")
    
    if label_stats:
        print(f"\n标签分布:")
        total_labeled = sum(label_stats.values())
        for label_type, count in label_stats.items():
            percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
            print(f"  {label_type}: {count} ({percentage:.2f}%)")
    
    return flows, host_flows

def analyze_host_level_distribution(host_flows):
    """分析主机级别的标签分布"""
    print(f"\n{'='*80}")
    print("主机级别标签分布分析")
    print(f"{'='*80}")
    
    host_analysis = defaultdict(lambda: {'normal': 0, 'anomaly': 0, 'total': 0})
    
    for host_ip, flows in host_flows.items():
        for flow in flows:
            if flow['label'] is not None:
                if flow['label'] == 0:
                    host_analysis[host_ip]['normal'] += 1
                elif flow['label'] == 1:
                    host_analysis[host_ip]['anomaly'] += 1
                host_analysis[host_ip]['total'] += 1
    
    # 分类主机
    pure_normal_hosts = []
    pure_anomaly_hosts = []
    mixed_label_hosts = []
    unlabeled_hosts = []
    
    for host_ip, counts in host_analysis.items():
        if counts['total'] == 0:
            unlabeled_hosts.append(host_ip)
        elif counts['anomaly'] > 0 and counts['normal'] == 0:
            pure_anomaly_hosts.append((host_ip, counts))
        elif counts['normal'] > 0 and counts['anomaly'] == 0:
            pure_normal_hosts.append((host_ip, counts))
        else:
            mixed_label_hosts.append((host_ip, counts))
    
    print(f"主机总数: {len(host_analysis)}")
    print(f"纯正常主机: {len(pure_normal_hosts)}")
    print(f"纯异常主机: {len(pure_anomaly_hosts)}")
    print(f"混合标签主机: {len(mixed_label_hosts)}")
    print(f"无标签主机: {len(unlabeled_hosts)}")
    
    # 分析混合标签主机
    if mixed_label_hosts:
        print(f"\n混合标签主机详细分析 (显示前10个):")
        mixed_label_hosts_sorted = sorted(mixed_label_hosts, 
                                         key=lambda x: x[1]['anomaly']/(x[1]['total']+1e-10), 
                                         reverse=True)
        
        for i, (host_ip, counts) in enumerate(mixed_label_hosts_sorted[:10]):
            anomaly_ratio = counts['anomaly'] / counts['total'] * 100
            print(f"  主机 {host_ip}: 总流量={counts['total']}, 正常={counts['normal']}, 异常={counts['anomaly']}, 异常占比={anomaly_ratio:.1f}%")
    
    # 纯异常主机详情
    if pure_anomaly_hosts:
        print(f"\n纯异常主机详情 (显示前5个):")
        for i, (host_ip, counts) in enumerate(pure_anomaly_hosts[:5]):
            print(f"  主机 {host_ip}: 异常流量数={counts['anomaly']}")
    
    return host_analysis, pure_normal_hosts, pure_anomaly_hosts, mixed_label_hosts

def visualize_label_distribution(flows, host_analysis, dataset_name=""):
    """可视化标签分布"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name} - 数据分布分析', fontsize=16, fontweight='bold')
    
    # 1. 流量级别标签分布
    if flows:
        labels = [f['label'] for f in flows if f['label'] is not None]
        if labels:
            label_counts = Counter(labels)
            label_names = ['正常' if l == 0 else '异常' for l in label_counts.keys()]
            values = list(label_counts.values())
            
            axes[0, 0].pie(values, labels=label_names, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('流量级别标签分布')
    
    # 2. 主机类别分布
    if host_analysis:
        host_categories = []
        for host_ip, counts in host_analysis.items():
            if counts['total'] == 0:
                host_categories.append('无标签')
            elif counts['anomaly'] > 0 and counts['normal'] == 0:
                host_categories.append('纯异常')
            elif counts['normal'] > 0 and counts['anomaly'] == 0:
                host_categories.append('纯正常')
            else:
                host_categories.append('混合标签')
        
        category_counts = Counter(host_categories)
        axes[0, 1].bar(category_counts.keys(), category_counts.values())
        axes[0, 1].set_title('主机类别分布')
        axes[0, 1].set_ylabel('主机数量')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 协议分布
    if flows:
        protocols = [f['protocol'] for f in flows]
        protocol_counts = Counter(protocols)
        
        # 映射协议号到名称
        protocol_names = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        protocol_labels = [protocol_names.get(p, f'协议{p}') for p in protocol_counts.keys()]
        
        axes[0, 2].bar(protocol_labels, protocol_counts.values())
        axes[0, 2].set_title('协议分布')
        axes[0, 2].set_ylabel('流量数量')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 异常流量按主机分布
    if host_analysis:
        anomaly_counts = []
        normal_counts = []
        host_ids = []
        
        # 只取有异常的主机
        anomaly_hosts = [(host, counts) for host, counts in host_analysis.items() 
                        if counts['anomaly'] > 0]
        anomaly_hosts_sorted = sorted(anomaly_hosts, key=lambda x: x[1]['anomaly'], reverse=True)[:10]
        
        for host_ip, counts in anomaly_hosts_sorted:
            host_ids.append(str(host_ip))
            anomaly_counts.append(counts['anomaly'])
            normal_counts.append(counts['normal'])
        
        if anomaly_counts:
            x = np.arange(len(host_ids))
            width = 0.35
            axes[1, 0].bar(x - width/2, normal_counts, width, label='正常流量')
            axes[1, 0].bar(x + width/2, anomaly_counts, width, label='异常流量')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(host_ids, rotation=45)
            axes[1, 0].set_title('异常主机流量分布 (前10个)')
            axes[1, 0].set_ylabel('流量数量')
            axes[1, 0].legend()
    
    # 5. 端口分布
    if flows:
        dst_ports = [f['dst_port'] for f in flows]
        # 只取最常见的前10个端口
        port_counts = Counter(dst_ports)
        common_ports = port_counts.most_common(10)
        
        if common_ports:
            ports, counts = zip(*common_ports)
            axes[1, 1].bar([f'端口{p}' for p in ports], counts)
            axes[1, 1].set_title('目标端口分布 (前10个)')
            axes[1, 1].set_ylabel('流量数量')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. 流量时间序列（模拟）
    if flows:
        # 假设每个流量按顺序到达
        anomaly_sequence = [1 if f.get('label') == 1 else 0 for f in flows[:200] if f.get('label') is not None]
        if anomaly_sequence:
            axes[1, 2].plot(range(len(anomaly_sequence)), anomaly_sequence, 'r-', linewidth=1)
            axes[1, 2].fill_between(range(len(anomaly_sequence)), anomaly_sequence, alpha=0.3, color='red')
            axes[1, 2].set_title('异常流量序列 (前200个流量)')
            axes[1, 2].set_xlabel('流量序列')
            axes[1, 2].set_ylabel('异常(1)/正常(0)')
            axes[1, 2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()

def analyze_majority_voting_issues(host_flows, threshold=0.5):
    """分析多数投票可能引起的问题"""
    print(f"\n{'='*80}")
    print("多数投票问题分析 (阈值={})".format(threshold))
    print(f"{'='*80}")
    
    problematic_hosts = []
    
    for host_ip, flows in host_flows.items():
        labels = [f['label'] for f in flows if f['label'] is not None]
        if not labels:
            continue
            
        normal_count = sum(1 for l in labels if l == 0)
        anomaly_count = sum(1 for l in labels if l == 1)
        total = len(labels)
        
        anomaly_ratio = anomaly_count / total
        
        # 多数投票结果
        majority_vote = 0 if normal_count > anomaly_count else 1
        
        # 检查问题
        problems = []
        if anomaly_count > 0 and majority_vote == 0:
            problems.append(f"有{anomaly_count}个异常流量但被多数投票标记为正常")
        if anomaly_ratio >= threshold and majority_vote == 0:
            problems.append(f"异常比例{anomaly_ratio:.1%}超过阈值{threshold}但仍标记为正常")
        
        if problems:
            problematic_hosts.append((host_ip, normal_count, anomaly_count, anomaly_ratio, majority_vote, problems))
    
    print(f"发现问题的主机数: {len(problematic_hosts)}")
    
    if problematic_hosts:
        print("\n问题主机详情 (显示前10个):")
        for i, (host_ip, normal, anomaly, ratio, vote, problems) in enumerate(problematic_hosts[:10]):
            print(f"  主机 {host_ip}: 正常={normal}, 异常={anomaly}, 异常比例={ratio:.1%}, 多数投票={vote}")
            for problem in problems:
                print(f"    ⚠️ {problem}")
    
    return problematic_hosts

def suggest_labeling_strategy(host_analysis):
    """根据分析结果建议标签策略"""
    print(f"\n{'='*80}")
    print("标签策略建议")
    print(f"{'='*80}")
    
    # 统计不同策略下的标签分布
    strategies = {
        "多数投票": defaultdict(int),
        "严格策略(任何异常即标记)": defaultdict(int),
        "阈值策略(异常比例>10%即标记)": defaultdict(int),
        "阈值策略(异常比例>30%即标记)": defaultdict(int),
    }
    
    for host_ip, counts in host_analysis.items():
        if counts['total'] == 0:
            continue
            
        normal = counts['normal']
        anomaly = counts['anomaly']
        total = counts['total']
        anomaly_ratio = anomaly / total
        
        # 多数投票
        strategies["多数投票"][0 if normal > anomaly else 1] += 1
        
        # 严格策略：任何异常即标记为异常
        strategies["严格策略(任何异常即标记)"][1 if anomaly > 0 else 0] += 1
        
        # 阈值策略
        strategies["阈值策略(异常比例>10%即标记)"][1 if anomaly_ratio > 0.1 else 0] += 1
        strategies["阈值策略(异常比例>30%即标记)"][1 if anomaly_ratio > 0.3 else 0] += 1
    
    print("不同标签策略下的主机分布:")
    for strategy_name, label_dist in strategies.items():
        normal_count = label_dist.get(0, 0)
        anomaly_count = label_dist.get(1, 0)
        total = normal_count + anomaly_count
        
        if total > 0:
            normal_pct = normal_count / total * 100
            anomaly_pct = anomaly_count / total * 100
            print(f"\n  {strategy_name}:")
            print(f"    正常主机: {normal_count} ({normal_pct:.1f}%)")
            print(f"    异常主机: {anomaly_count} ({anomaly_pct:.1f}%)")
    
    # 推荐策略
    print(f"\n推荐策略:")
    print("  1. 如果异常检测对误报不敏感，建议使用'严格策略(任何异常即标记)'")
    print("  2. 如果希望平衡误报和漏报，建议使用'阈值策略(异常比例>30%即标记)'")
    print("  3. 如果数据质量高，异常流量明显，可以使用'多数投票'")

def analyze_for_svm_training(host_analysis, host_flows):
    """为SVM训练准备的分析"""
    print(f"\n{'='*80}")
    print("SVM训练准备分析")
    print(f"{'='*80}")
    
    # 使用严格策略生成标签
    y_labels = []
    host_ids = []
    
    for host_ip, counts in host_analysis.items():
        if counts['total'] == 0:
            continue  # 跳过无标签主机
        
        # 严格策略：任何异常即标记为异常
        label = 1 if counts['anomaly'] > 0 else 0
        y_labels.append(label)
        host_ids.append(host_ip)
    
    y = np.array(y_labels)
    
    print(f"可用于SVM训练的主机数: {len(y)}")
    print(f"标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}")
    
    if len(np.unique(y)) < 2:
        print("\n⚠️ 警告: 只有一个类别，SVM无法训练！")
        print("建议:")
        print("  1. 检查原始数据中是否有异常流量")
        print("  2. 调整标签策略，确保至少有一个异常主机")
        print("  3. 如果确实没有异常，考虑生成模拟数据")
    else:
        print("\n✓ 数据包含两个类别，可以用于SVM训练")
        
        # 检查类别不平衡
        imbalance_ratio = np.sum(y==1) / len(y)
        if imbalance_ratio < 0.1:
            print(f"⚠️ 警告: 类别严重不平衡，异常类仅占 {imbalance_ratio:.1%}")
            print("建议使用以下技术:")
            print("  1. 过采样 (SMOTE)")
            print("  2. 调整类别权重 (class_weight='balanced')")
            print("  3. 使用合适的评估指标 (F1-score, ROC-AUC)")
    
    return y, host_ids

def main():
    """主函数"""
    print("网络异常检测数据诊断工具")
    print("="*80)
    
    # 设置数据文件路径
    data_dir = './data'
    labeled_file = os.path.join(data_dir, 'annotated-trace.csv')
    unlabeled_file = os.path.join(data_dir, 'not-annotated-trace.csv')
    
    # 检查文件是否存在
    if not os.path.exists(labeled_file):
        print(f"警告: 标注数据文件不存在: {labeled_file}")
        print("请确保数据文件在正确位置，或修改文件路径。")
        return
    
    # 1. 加载并分析标注数据
    labeled_flows, labeled_hosts = load_and_analyze_data(labeled_file, "标注数据集")
    
    if not labeled_flows:
        print("没有加载到标注数据，退出分析。")
        return
    
    # 2. 主机级别分析
    host_analysis, pure_normal_hosts, pure_anomaly_hosts, mixed_label_hosts = analyze_host_level_distribution(labeled_hosts)
    
    # 3. 可视化
    visualize_label_distribution(labeled_flows, host_analysis, "标注数据集")
    
    # 4. 分析多数投票问题
    problematic_hosts = analyze_majority_voting_issues(labeled_hosts, threshold=0.1)
    
    # 5. 建议标签策略
    suggest_labeling_strategy(host_analysis)
    
    # 6. SVM训练准备分析
    y_labels, host_ids = analyze_for_svm_training(host_analysis, labeled_hosts)
    
    # 7. 加载并分析未标注数据（如果存在）
    if os.path.exists(unlabeled_file):
        unlabeled_flows, unlabeled_hosts = load_and_analyze_data(unlabeled_file, "未标注数据集")
        
        if unlabeled_flows:
            print(f"\n{'='*80}")
            print("未标注数据集统计摘要")
            print(f"{'='*80}")
            print(f"未标注流量数: {len(unlabeled_flows)}")
            print(f"未标注主机数: {len(unlabeled_hosts)}")
            
            # 分析未标注主机的流量模式
            print("\n未标注主机流量统计 (前5个):")
            for i, (host_ip, flows) in enumerate(list(unlabeled_hosts.items())[:5]):
                print(f"  主机 {host_ip}: {len(flows)} 个流量")
    
    print(f"\n{'='*80}")
    print("分析完成!")
    print(f"{'='*80}")
    
    # 生成总结报告
    print("\n总结报告:")
    print(f"1. 标注数据集: {len(labeled_flows)} 条流量，{len(labeled_hosts)} 个主机")
    print(f"2. 异常流量主要集中在 {len(pure_anomaly_hosts) + len(mixed_label_hosts)} 个主机")
    print(f"3. {len(problematic_hosts)} 个主机可能存在多数投票问题")
    
    if len(pure_anomaly_hosts) + len(mixed_label_hosts) == 0:
        print("\n⚠️ 严重问题: 没有发现异常主机!")
        print("  这会导致SVM训练失败。")
        print("  建议检查数据文件，确保异常标签被正确解析。")

if __name__ == "__main__":
    main()