import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
import warnings
import os

warnings.filterwarnings('ignore')


def manual_graphlet_example():
    print("=" * 60)
    print("Manual Graphlet Construction Example (CSV格式)")
    print("=" * 60)

    # 使用CSV格式的示例数据
    example_flows = [
        "261,264,17,138,138,normal",  # CSV格式: srcIP,dstIP,protocol,sPort,dPort,label
        "261,265,17,80,80,normal",
        "262,266,6,167,80,normal",
        "263,264,6,443,443,normal",
        "261,265,17,443,80,normal"
    ]

    print("CSV格式的示例流量数据:")
    for flow in example_flows:
        print(f"  {flow}")

    from graphlet_builder import GraphletConstructor

    builder = GraphletConstructor()

    print("\n从CSV数据构建Graphlet:")
    for flow in example_flows:
        builder.add_flow_to_host(*builder.parse_network_flow(flow))

    builder.display_graphlet(261)

    return builder


def svm_without_kernel():
    print("=" * 60)
    print("Experiment 1: SVM Without Kernel Trick")
    print("=" * 60)

    from graphlet_builder import GraphletConstructor
    from random_walk_kernel import WalkKernelExtractor

    constructor = GraphletConstructor()

    # 从data目录读取标注的CSV数据
    labeled_file = "../data/annotated-trace.csv"
    if not os.path.exists(labeled_file):
        print(f"Error: File {labeled_file} not found!")
        print("Please place your CSV data file in the data/ directory.")
        return None, None, 0

    print(f"Loading labeled data from {labeled_file}")
    constructor.load_csv_data(labeled_file, has_labels=True)

    kernel_extractor = WalkKernelExtractor(walk_steps=4)

    hosts_list = list(constructor.host_graphlets.keys())[:50]
    feature_vectors = []
    host_labels = []

    print("Extracting features...")
    start_time = time.time()

    for host in hosts_list:
        profile_graph = constructor.create_profile_graphlet(host)
        walks = kernel_extractor.generate_walks(profile_graph, max_walk_count=200)
        features = kernel_extractor.convert_walks_to_features(walks, use_hashing=False)
        feature_vectors.append(features)

        labels_for_host = [flow['label'] for flow in constructor.host_flows[host] if flow['label'] is not None]
        if labels_for_host:
            dominant_label = max(set(labels_for_host), key=labels_for_host.count)
            host_labels.append(dominant_label)
        else:
            host_labels.append(0)

    max_feature_length = max(len(f) for f in feature_vectors)
    padded_features = [np.pad(f, (0, max_feature_length - len(f))) for f in feature_vectors]

    X = np.array(padded_features)
    y = np.array(host_labels)

    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time:.2f} seconds")
    print(f"Feature dimension: {X.shape[1]}")

    print("\nTraining SVM classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_start = time.time()
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)
    svm_time = time.time() - svm_start

    y_predicted = svm_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predicted)

    print(f"\nSVM training time: {svm_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_predicted, target_names=['Normal', 'Malicious']))

    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Without Kernel Trick)')
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Malicious'])
    plt.yticks([0, 1], ['Normal', 'Malicious'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    return svm_classifier, scaler, feature_time + svm_time


def svm_with_kernel():
    print("=" * 60)
    print("Experiment 2: SVM With Kernel Trick")
    print("=" * 60)

    from graphlet_builder import GraphletConstructor
    from random_walk_kernel import WalkKernelExtractor

    constructor = GraphletConstructor()

    # 从data目录读取标注的CSV数据
    labeled_file = "../data/annotated-trace.csv"
    if not os.path.exists(labeled_file):
        print(f"Error: File {labeled_file} not found!")
        print("Please place your CSV data file in the data/ directory.")
        return None, 0

    print(f"Loading labeled data from {labeled_file}")
    constructor.load_csv_data(labeled_file, has_labels=True)

    kernel_extractor = WalkKernelExtractor(walk_steps=4)

    hosts_list = list(constructor.host_graphlets.keys())[:50]
    graphlets_collection = []
    host_labels = []

    print("Building kernel matrix...")
    kernel_start = time.time()

    for host in hosts_list:
        profile_graph = constructor.create_profile_graphlet(host)
        graphlets_collection.append(profile_graph)

        labels_for_host = [flow['label'] for flow in constructor.host_flows[host] if flow['label'] is not None]
        if labels_for_host:
            dominant_label = max(set(labels_for_host), key=labels_for_host.count)
            host_labels.append(dominant_label)
        else:
            host_labels.append(0)

    K = kernel_extractor.compute_kernel_matrix(graphlets_collection)
    kernel_time = time.time() - kernel_start

    print(f"Kernel matrix computed in {kernel_time:.2f} seconds")
    print(f"Kernel matrix shape: {K.shape}")

    # 验证核矩阵是否是方阵
    if K.shape[0] != K.shape[1]:
        print(f"Warning: Kernel matrix is not square! Shape: {K.shape}")
        min_dim = min(K.shape[0], K.shape[1])
        K = K[:min_dim, :min_dim]
        host_labels = host_labels[:min_dim]
        print(f"Adjusted shape to: {K.shape}")

    print("\nTraining kernel SVM...")

    n_samples = len(host_labels)

    indices = np.arange(n_samples)
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, np.array(host_labels), test_size=0.3, random_state=42
    )

    K_train = K[np.ix_(train_idx, train_idx)]
    K_test = K[np.ix_(test_idx, train_idx)]

    svm_start = time.time()
    kernel_svm_model = SVC(kernel='precomputed', random_state=42)
    kernel_svm_model.fit(K_train, y_train)
    svm_time = time.time() - svm_start

    y_predicted = kernel_svm_model.predict(K_test)
    accuracy = accuracy_score(y_test, y_predicted)

    print(f"\nKernel SVM training time: {svm_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_predicted, target_names=['Normal', 'Malicious']))

    cm = confusion_matrix(y_test, y_predicted)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (With Kernel Trick)')
    plt.colorbar()
    plt.xticks([0, 1], ['Normal', 'Malicious'])
    plt.yticks([0, 1], ['Normal', 'Malicious'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    print("\nKernel Trick Summary:")
    print(f"  - Kernel matrix computation: {kernel_time:.2f} seconds")
    print(f"  - SVM training: {svm_time:.2f} seconds")
    print(f"  - Total time: {kernel_time + svm_time:.2f} seconds")
    print(f"  - Kernel matrix size: {K.shape[0]} x {K.shape[1]}")
    print(f"  - Training samples: {len(train_idx)}")
    print(f"  - Test samples: {len(test_idx)}")

    return kernel_svm_model, kernel_time + svm_time


def detect_anomalies():
    print("=" * 60)
    print("Experiment 3: Anomaly Detection in Unlabeled Data")
    print("=" * 60)

    from graphlet_builder import GraphletConstructor
    from random_walk_kernel import WalkKernelExtractor

    # 1. 加载未标注数据
    constructor = GraphletConstructor()
    unlabeled_file = "../data/not-annotated-trace.csv"

    if not os.path.exists(unlabeled_file):
        print(f"Error: File {unlabeled_file} not found!")
        print("Please place your unlabeled CSV data file in the data/ directory.")
        return [], []

    print(f"Loading unlabeled data from {unlabeled_file}")
    constructor.load_csv_data(unlabeled_file, has_labels=False)

    print("\nTraining model on labeled data...")

    # 2. 使用标注数据训练模型
    training_builder = GraphletConstructor()
    labeled_file = "../data/annotated-trace.csv"

    if not os.path.exists(labeled_file):
        print(f"Error: File {labeled_file} not found!")
        print("Please place your labeled CSV data file in the data/ directory.")
        return [], []

    print(f"Loading training data from {labeled_file}")
    training_builder.load_csv_data(labeled_file, has_labels=True)

    kernel_extractor = WalkKernelExtractor(walk_steps=4)
    hosts = list(training_builder.host_graphlets.keys())[:40]

    X_train = []
    y_train = []

    for host in hosts:
        profile_graph = training_builder.create_profile_graphlet(host)
        walks = kernel_extractor.generate_walks(profile_graph, max_walk_count=200)
        features = kernel_extractor.convert_walks_to_features(walks, use_hashing=True, feature_dim=512)
        X_train.append(features)

        host_labels = [flow['label'] for flow in training_builder.host_flows[host] if flow['label'] is not None]
        if host_labels:
            dominant_label = max(set(host_labels), key=host_labels.count)
            y_train.append(dominant_label)
        else:
            y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)

    print(f"Model trained with {len(X_train)} samples")

    print("\nDetecting anomalies in unlabeled data...")

    suspicious_hosts_list = []
    anomaly_scores_list = []

    for host in constructor.host_graphlets.keys():
        profile_graph = constructor.create_profile_graphlet(host)
        walks = kernel_extractor.generate_walks(profile_graph, max_walk_count=200)
        features = kernel_extractor.convert_walks_to_features(walks, use_hashing=True, feature_dim=512)

        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        elif len(features) > 512:
            features = features[:512]

        prediction = svm.predict([features])[0]
        decision_value = svm.decision_function([features])[0]

        if prediction == 1:
            suspicious_hosts_list.append(host)
            anomaly_scores_list.append(abs(decision_value))

    print(f"\nDetection Results:")
    print(f"Detected {len(suspicious_hosts_list)} suspicious hosts")

    if suspicious_hosts_list:
        print("\nSuspicious hosts (sorted by anomaly score):")
        for host, score in sorted(zip(suspicious_hosts_list, anomaly_scores_list), key=lambda x: x[1], reverse=True):
            print(f"  Host {host}: anomaly score = {score:.4f}")

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(suspicious_hosts_list)), sorted(anomaly_scores_list, reverse=True))
        plt.xlabel('Suspicious Hosts (sorted by anomaly)')
        plt.ylabel('Anomaly Score')
        plt.title('Anomaly Score Distribution')

        threshold_value = np.mean(anomaly_scores_list) + np.std(anomaly_scores_list)
        for i, bar in enumerate(bars):
            if bar.get_height() > threshold_value:
                bar.set_color('red')

        plt.axhline(y=threshold_value, color='r', linestyle='--', label=f'Threshold ({threshold_value:.2f})')
        plt.legend()
        plt.xticks([])
        plt.tight_layout()
        plt.show()

        print("\nAttack type analysis:")
        for host in suspicious_hosts_list[:3]:
            print(f"\nAnalyzing host {host}:")
            flows = constructor.host_flows.get(host, [])

            if len(flows) > 0:
                destination_ports = [flow['destination_port'] for flow in flows]
                unique_ports_count = len(set(destination_ports))
                total_flow_count = len(flows)

                print(f"  Total flows: {total_flow_count}")
                print(f"  Unique destination ports: {unique_ports_count}")

                if unique_ports_count > 20 and total_flow_count > 30:
                    print(f"  ⚠️  Potential port scan (scanned {unique_ports_count} different ports)")

                if total_flow_count > 100:
                    destination_ips = [flow['destination_ip'] for flow in flows]
                    unique_destination_count = len(set(destination_ips))
                    if unique_destination_count < 5 and total_flow_count > 100:
                        print(
                            f"  ⚠️  Potential DDoS attack ({total_flow_count} connections to {unique_destination_count} targets)")
    else:
        print("No suspicious hosts detected")

    return suspicious_hosts_list, anomaly_scores_list


def analyze_false_results():
    print("=" * 60)
    print("Experiment 4: False Positive and False Negative Analysis")
    print("=" * 60)

    from graphlet_builder import GraphletConstructor
    from random_walk_kernel import WalkKernelExtractor

    constructor = GraphletConstructor()

    labeled_file = "../data/annotated-trace.csv"
    if not os.path.exists(labeled_file):
        print(f"Error: File {labeled_file} not found!")
        print("Please place your CSV data file in the data/ directory.")
        return None

    print(f"Loading data from {labeled_file}")
    constructor.load_csv_data(labeled_file, has_labels=True)

    kernel_extractor = WalkKernelExtractor(walk_steps=4)
    hosts = list(constructor.host_graphlets.keys())[:30]

    X = []
    y_actual = []

    for host in hosts:
        profile_graph = constructor.create_profile_graphlet(host)
        walks = kernel_extractor.generate_walks(profile_graph, max_walk_count=200)
        features = kernel_extractor.convert_walks_to_features(walks, use_hashing=True, feature_dim=256)

        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))

        X.append(features)

        host_labels = [flow['label'] for flow in constructor.host_flows[host] if flow['label'] is not None]
        if host_labels:
            dominant_label = max(set(host_labels), key=host_labels.count)
            y_actual.append(dominant_label)
        else:
            y_actual.append(0)

    X = np.array(X)
    y_actual = np.array(y_actual)

    svm = SVC(kernel='linear', random_state=42)
    y_predicted = cross_val_predict(svm, X, y_actual, cv=5)

    cm = confusion_matrix(y_actual, y_predicted)
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix:")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")

    precision_value = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_value = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (
                                                                                                    precision_value + recall_value) > 0 else 0

    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision_value:.4f}")
    print(f"  Recall: {recall_value:.4f}")
    print(f"  F1 Score: {f1_value:.4f}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    matrix_labels = ['TN', 'FP', 'FN', 'TP']
    matrix_values = [tn, fp, fn, tp]
    matrix_colors = ['green', 'red', 'orange', 'blue']
    plt.bar(matrix_labels, matrix_values, color=matrix_colors)
    plt.title('Confusion Matrix Values')
    plt.ylabel('Count')

    for i, v in enumerate(matrix_values):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.subplot(1, 2, 2)
    metric_names = ['Precision', 'Recall', 'F1 Score']
    metric_values = [precision_value, recall_value, f1_value]
    metric_colors = ['purple', 'cyan', 'magenta']
    plt.bar(metric_names, metric_values, color=metric_colors)
    plt.title('Performance Metrics')
    plt.ylim(0, 1)

    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.show()

    return cm


def advanced_non_tottering_method():
    print("=" * 60)
    print("Experiment 5: Advanced Method Without Tottering Walks")
    print("=" * 60)

    from graphlet_builder import GraphletConstructor
    from random_walk_kernel import WalkKernelExtractor, AdvancedWalkKernel

    constructor = GraphletConstructor()

    labeled_file = "../data/annotated-trace.csv"
    if not os.path.exists(labeled_file):
        print(f"Error: File {labeled_file} not found!")
        print("Please place your CSV data file in the data/ directory.")
        return None, None

    print(f"Loading data from {labeled_file}")
    constructor.load_csv_data(labeled_file, has_labels=True)

    test_host = list(constructor.host_graphlets.keys())[0]
    test_graphlet = constructor.create_profile_graphlet(test_host)

    print("Comparing basic and advanced methods:")

    print("\n1. Basic random walk method:")
    basic_kernel = WalkKernelExtractor(walk_steps=4)
    start_time = time.time()
    basic_walks = basic_kernel.generate_walks(test_graphlet, max_walk_count=100)
    basic_time = time.time() - start_time
    print(f"   Extracted {len(basic_walks)} walks")
    print(f"   Time: {basic_time:.4f} seconds")

    tottering_count_basic = 0
    for walk in basic_walks:
        for i in range(len(walk) - 1):
            if walk[i] == walk[i + 1]:
                tottering_count_basic += 1

    print(f"   Detected {tottering_count_basic} tottering occurrences")

    print("\n2. Advanced non-tottering method:")
    advanced_kernel = AdvancedWalkKernel(walk_steps=4)
    start_time = time.time()
    advanced_walks = advanced_kernel.generate_walks(test_graphlet, max_walk_count=100)
    advanced_time = time.time() - start_time
    print(f"   Extracted {len(advanced_walks)} walks")
    print(f"   Time: {advanced_time:.4f} seconds")

    tottering_count_advanced = 0
    for walk in advanced_walks:
        for i in range(len(walk) - 1):
            if walk[i] == walk[i + 1]:
                tottering_count_advanced += 1

    print(f"   Detected {tottering_count_advanced} tottering occurrences")

    print("\n3. Classification performance comparison:")

    hosts = list(constructor.host_graphlets.keys())[:20]

    X_basic = []
    y_labels = []

    for host in hosts:
        profile_graph = constructor.create_profile_graphlet(host)
        walks = basic_kernel.generate_walks(profile_graph, max_walk_count=100)
        features = basic_kernel.convert_walks_to_features(walks, use_hashing=True, feature_dim=256)

        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))

        X_basic.append(features)

        host_labels = [flow['label'] for flow in constructor.host_flows[host] if flow['label'] is not None]
        if host_labels:
            dominant_label = max(set(host_labels), key=host_labels.count)
            y_labels.append(dominant_label)
        else:
            y_labels.append(0)

    X_basic = np.array(X_basic)
    y_labels = np.array(y_labels)

    X_advanced = []

    for host in hosts:
        profile_graph = constructor.create_profile_graphlet(host)
        walks = advanced_kernel.generate_walks(profile_graph, max_walk_count=100)
        features = advanced_kernel.convert_walks_to_features(walks, use_hashing=True, feature_dim=256)

        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))

        X_advanced.append(features)

    X_advanced = np.array(X_advanced)

    svm = SVC(kernel='linear', random_state=42)

    print("\n   Basic method cross-validation accuracy:")
    basic_scores = cross_val_score(svm, X_basic, y_labels, cv=3)
    print(f"    {basic_scores.mean():.4f} (+/- {basic_scores.std() * 2:.4f})")

    print("\n   Advanced method cross-validation accuracy:")
    advanced_scores = cross_val_score(svm, X_advanced, y_labels, cv=3)
    print(f"    {advanced_scores.mean():.4f} (+/- {advanced_scores.std() * 2:.4f})")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.bar(['Basic', 'Advanced'], [basic_time, advanced_time], color=['blue', 'green'])
    plt.title('Computation Time Comparison')
    plt.ylabel('Time (seconds)')

    for i, v in enumerate([basic_time, advanced_time]):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center')

    plt.subplot(1, 3, 2)
    plt.bar(['Basic', 'Advanced'], [tottering_count_basic, tottering_count_advanced], color=['blue', 'green'])
    plt.title('Tottering Occurrences Comparison')
    plt.ylabel('Count')

    for i, v in enumerate([tottering_count_basic, tottering_count_advanced]):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.subplot(1, 3, 3)
    x_positions = np.arange(2)
    plt.bar(x_positions - 0.2, [basic_scores.mean(), advanced_scores.mean()],
            width=0.4, color=['blue', 'green'], label='Mean Accuracy')
    plt.errorbar(x_positions - 0.2, [basic_scores.mean(), advanced_scores.mean()],
                 yerr=[basic_scores.std(), advanced_scores.std()],
                 fmt='o', color='black', capsize=5)
    plt.title('Classification Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(x_positions, ['Basic', 'Advanced'])
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

    return basic_kernel, advanced_kernel


def main_execution():
    print("=" * 60)
    print("Network Traffic Anomaly Detection Project")
    print("Graphlet and SVM Based Approach")
    print("=" * 60)

    print("\nExecuting Experiment 0: Manual Graphlet Construction")
    graphlet_builder = manual_graphlet_example()

    print("\nExecuting Experiment 1: Without Kernel Trick")
    svm_no_kernel_model, scaler, time_no_kernel = svm_without_kernel()

    print("\nExecuting Experiment 2: With Kernel Trick")
    svm_kernel_model, time_with_kernel = svm_with_kernel()

    print("\n" + "=" * 60)
    print("Computation Time Comparison:")
    print(f"  Without kernel trick: {time_no_kernel:.2f} seconds")
    print(f"  With kernel trick: {time_with_kernel:.2f} seconds")
    print(f"  Time difference: {abs(time_no_kernel - time_with_kernel):.2f} seconds")
    print("=" * 60)

    print("\nExecuting Experiment 3: Anomaly Detection in Unlabeled Data")
    suspicious_hosts, anomaly_scores = detect_anomalies()

    print("\nExecuting Experiment 4: False Positive/Negative Analysis")
    confusion_matrix_result = analyze_false_results()

    print("\nExecuting Experiment 5: Advanced Non-Tottering Method")
    basic_kernel, advanced_kernel = advanced_non_tottering_method()

    print("\n" + "=" * 60)
    print("Project Summary")
    print("=" * 60)
    print("Completed tasks:")
    print("1. Graphlet construction and understanding")
    print("2. Feature extraction using random walk kernels")
    print("3. SVM classification with and without kernel trick")
    print("4. Anomaly detection in unlabeled traffic")
    print("5. False positive/negative analysis")
    print("6. Advanced method without tottering walks")
    print("=" * 60)

    return {
        'graphlet_constructor': graphlet_builder,
        'svm_no_kernel_model': svm_no_kernel_model,
        'svm_kernel_model': svm_kernel_model,
        'suspicious_hosts': suspicious_hosts,
        'confusion_matrix': confusion_matrix_result
    }


if __name__ == "__main__":
    final_results = main_execution()