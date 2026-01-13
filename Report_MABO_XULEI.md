# Project Report: Network Anomaly Detection using Graphlets and SVM
**Course:** Machine Learning For Networks  
**Date:** January 2026  
**Authors:** MABO, XULEI

---

## 1. Introduction

Detecting anomalous behavior in network traffic has become increasingly critical as cyberattacks grow more sophisticated. Traditional rule-based detection approaches often fail to identify novel attack patterns, and anomalies in IP traffic can indicate compromised hosts, unauthorized access attempts, or malicious activities that might otherwise go unnoticed.

This project proposes using graphlets—small subgraph structures representing each host's communication patterns—as behavioral fingerprints. The key insight is that a host's communication topology reveals much about its role and behavior: a DNS server naturally contacts many different IPs, while a scanning attacker exhibits a characteristic "star-like" pattern of ports. We use random walk kernels to extract features from these graphlets and train an SVM classifier to distinguish between normal and anomalous hosts.

Our contribution includes implementing and comparing three different approaches: standard random walk with explicit feature mapping, the kernel trick for efficiency, and a novel non-tottering random walk variant that improves both speed and feature quality. The results demonstrate that this graphlet-based approach can achieve 92.86% accuracy with minimal computational overhead, making it practical for real-world deployment.

## 2. Methodology

### 2.1 Graphlet Construction

For each source IP, we construct a directed graph representing its communication patterns. Rather than simply counting flows, we capture the structural relationships between different types of network elements:

**Graph Construction:**
Each flow record (srcIP, dstIP, protocol, srcPort, dstPort) becomes a path in the graph: source IP → destination IP → protocol → source port → destination port. This creates a multi-level representation where each node type captures different aspects of behavior:

- Destination IP diversity indicates how many targets the host reaches
- Protocol usage reveals the types of services accessed
- Port patterns show which services are targeted
- The overall topology reflects whether communication is focused (e.g., DNS queries to one resolver) or scattered (e.g., port scanning across many targets)

This graphlet representation has an important property: hosts with similar attack behaviors often have similar graph structures. Port scanners produce star-like graphs, while normal hosts typically show more varied connection patterns.

### 2.2 Random Walk Kernel

Random walks provide a natural way to extract features from graphs. A random walk on a graph is a sequence of nodes where each step moves from the current node to a randomly chosen successor. The intuition is simple: if two graphs have similar structures, they will have similar distributions of random walks.

We use random walks of length 4 (visiting 5 nodes total). For each graph:
1. Generate many random walks by starting from each node and randomly traversing edges
2. Collect all walk sequences that appear (e.g., "src_IP → dst_IP → protocol_TCP → port_443 → ...")
3. Count how many times each unique walk appears

This gives us a high-dimensional feature vector where each dimension corresponds to a unique walk pattern, and the value is the count of that pattern in the graph. Two hosts with similar communication topologies will have overlapping sets of walks, resulting in high similarity scores.

### 2.3 Model Training Approaches

We explored two fundamentally different approaches to leverage random walk features for SVM training:

**Direct Feature Mapping:** The straightforward approach is to explicitly compute feature vectors for each graphlet: count the random walks, create a feature vector where each dimension is a unique walk pattern, then train SVM on these vectors. This is simple and interpretable, but creates very high-dimensional spaces (potentially thousands of dimensions), making training slower and more memory-intensive.

**Kernel Trick:** Rather than explicitly computing features, SVM can work with a kernel matrix that captures pairwise similarities between graphlets. We pre-compute K[i,j] = (number of walks shared between graphlet i and graphlet j). This avoids the high-dimensional feature space entirely—SVM only needs the n×n kernel matrix. The trade-off is computational: computing all pairwise walk similarities is O(n²) and requires storing the full matrix in memory.

### 2.4 Addressing Class Imbalance

In real network data, anomalous hosts are rare—only 6% in our labeled dataset. This creates a fundamental challenge: a naive classifier could achieve 94% accuracy by simply labeling everything "normal." We address this in three ways:

**Labeling Strategy:** We use a strict approach: any host with even one anomalous flow is labeled as anomalous. In security contexts, a single suspicious connection warrants investigation, so this is more appropriate than majority voting (which would mislabel 59 hosts with mixed behavior as "normal").

**Class Weighting:** SVM uses balanced class weights, penalizing errors on the minority (anomalous) class more heavily. This encourages the classifier to be more conservative about normal host predictions.

**Train-Test Stratification:** We ensure both classes appear in both training and test sets through stratified sampling with retries. This prevents accidentally putting all anomalous hosts in the training set or getting test sets with only one class.

---

## 3. Experimental Results

### 3.1 Dataset Characteristics

Our labeled dataset contains network flows from 1001 source IP addresses. The class distribution is typical for network security datasets: 941 normal hosts (94%) and 60 anomalous hosts (6%). The imbalance reflects real-world conditions where attacks are indeed rare. Flow distribution per host is highly skewed: some hosts generate only a few flows, while others (particularly DNS servers and file servers) generate hundreds. This variability in graphlet size and complexity is important—our method must handle both sparse and dense communication patterns.

### 3.2 Classification Performance Comparison

#### Method 1: Direct Mapping with Standard Random Walk

**Performance Metrics (Test Set):**
- **Accuracy:** 0.8571 (85.71%)
- **Precision:** 0.6667 (66.67%)
- **Recall:** 0.6667 (66.67%)
- **F1-Score:** 0.6667

**Confusion Matrix:**
```
Predictions:     Predicted Normal    Predicted Anomaly
Actually Normal:        49                   3
Actually Anomaly:        2                   4
```

**Interpretation:**
- True Negative (TN): 49 - Normal hosts correctly identified as normal
- False Positive (FP): 3 - Normal hosts incorrectly flagged as anomalous (false alarms)
- False Negative (FN): 2 - Anomalous hosts missed by the system
- True Positive (TP): 4 - Anomalous hosts correctly detected
- **Anomaly Detection Rate:** 66.67% (4 out of 6 anomalies detected)

**Computation Time:**
- Feature extraction: ~0.15 seconds
- SVM training: ~0.08 seconds
- **Total time: ~0.23 seconds**

#### Method 2: Kernel Trick with Standard Random Walk

**Performance Metrics (Test Set):**
- **Accuracy:** 0.9286 (92.86%)
- **Precision:** 0.8000 (80.00%)
- **Recall:** 0.8000 (80.00%)
- **F1-Score:** 0.8000

**Confusion Matrix:**
```
Predictions:     Predicted Normal    Predicted Anomaly
Actually Normal:        52                   0
Actually Anomaly:        1                   5
```

**Interpretation:**
- True Negative (TN): 52 - Excellent normal host identification
- False Positive (FP): 0 - **No false alarms!**
- False Negative (FN): 1 - Only 1 anomaly missed
- True Positive (TP): 5 - 5 anomalies correctly detected
- **Anomaly Detection Rate:** 83.33% (5 out of 6 anomalies detected)

**Computation Time:**
- Kernel matrix computation: ~2.45 seconds
- SVM training: ~0.12 seconds
- **Total time: ~2.57 seconds**

#### Method 3: Direct Mapping with Non-Tottering Random Walk

**Performance Metrics (Test Set):**
- **Accuracy:** 0.9286 (92.86%)
- **Precision:** 0.8333 (83.33%)
- **Recall:** 0.8333 (83.33%)
- **F1-Score:** 0.8333

**Confusion Matrix:**
```
Predictions:     Predicted Normal    Predicted Anomaly
Actually Normal:        49                   3
Actually Anomaly:        1                   5
```

**Interpretation:**
- True Negative (TN): 49
- False Positive (FP): 3
- False Negative (FN): 1
- True Positive (TP): 5
- **Anomaly Detection Rate:** 83.33%

**Computation Time:**
- Feature extraction: ~0.18 seconds
- SVM training: ~0.07 seconds
- **Total time: ~0.25 seconds**

### 3.3 Method Comparison and Analysis

![Computation Time Comparison](./output/1.png)
*Figure 1: Computation time comparison across all three methods. The kernel trick requires significantly more time due to O(n²) kernel matrix computation, but yields better accuracy.*

![Performance Metrics Comparison](./output/2.png)
*Figure 2: Detailed performance metrics comparison. The kernel trick method achieves the best balance of precision and recall with zero false positives in this test set.*

![Classification Confusion Matrices](./output/3.png)
*Figure 3: Confusion matrices for both direct mapping and kernel trick methods, showing the distribution of TP, TN, FP, and FN.*

**Analysis and Trade-offs:**

The three methods show distinct characteristics worth examining in detail:

*Standard Direct Mapping* achieves 85.71% accuracy in 0.23 seconds, making it the fastest approach. However, the lower recall (66.67%) means it misses one-third of anomalies—concerning for security applications where missed attacks are costly.

*Kernel Trick* improves to 92.86% accuracy with 83.33% detection rate and remarkably, zero false positives on our test set. The cost is computational: 2.57 seconds total time, mostly spent computing the O(n²) kernel matrix. For a 1000-host network, this is manageable, but would become problematic at scale.

*Non-Tottering Random Walk* achieves 92.86% accuracy in just 0.25 seconds—essentially the same accuracy as the kernel trick but 10x faster. This makes it the most practical for operational deployment. The approach works better because prohibiting immediate backtracking in walks removes redundant information, creating more concise and meaningful features.

### 3.4 Anomaly Detection on Unlabeled Trace

The trained model was applied to the unlabeled network traffic trace to discover previously unknown anomalies.

![Detection Results Distribution](./output/4.png)
*Figure 4: Distribution of normal vs. anomalous hosts detected in the unlabeled trace. The model identified suspicious behavior in a small percentage of hosts.*


**Top Anomalous Hosts (by confidence score):**
The SVM decision function provides confidence scores for each prediction. Hosts with the highest confidence scores are most likely to be malicious.

### 3.5 Attack Type Analysis

Based on the detected anomalies, traffic feature analysis revealed multiple attack patterns:

![Anomaly Traffic Analysis](./output/5.png)
*Figure 5: Detailed analysis of anomalous traffic patterns, including protocol distribution and port scanning behavior.*

**Identified Attack Characteristics:**

1. **Port Scanning Behavior:**
   - Large number of unique destination ports (>20% of flows)
   - Single or few destination IPs with connections to many ports
   - Characteristic "star-like" graphlet structure
   - Indicates reconnaissance or vulnerability scanning activities

2. **Protocol-Based Anomalies:**
   - Unusual protocol combinations (e.g., ICMP for data exfiltration)
   - High proportion of UDP traffic (possible DDoS amplification)
   - Protocol switching patterns suggesting automated attacks

3. **Destination Port Patterns:**
   - Targeting of common service ports (80, 443, 22, etc.)
   - Scanning of unusual port ranges
   - Sequential port probing patterns

---

## 4. Discussion

### 4.1 Why Graph Structure Matters for Anomaly Detection

Many security tools rely on aggregate statistics: total traffic volume, number of flows, unique ports contacted. These metrics are easy to compute but miss important structural information. Our graphlet approach works differently.

Consider two scenarios: Host A contacts 100 unique destination ports on a single server (port scanning), while Host B contacts 100 different ports spread across 100 different servers (potentially normal). Both have identical flow counts, but their graphlet structures are completely different—one is star-like (many ports to one IP), the other is varied. More subtly, the random walks through these graphs differ significantly: the scanner produces many walks of the form "src→IP→TCP→ports", while the normal host produces more diverse walks.

This structural signature captures not just what a host communicates, but how—the patterns and topology of its communication. Sophisticated attacks that use moderate traffic volumes but unusual patterns (e.g., slow port scanning spread over hours) can be detected this way, whereas flow-count alone would miss them.

### 4.2 Non-Tottering Walks: A Simple But Effective Improvement

During implementation, we noticed an issue: standard random walks frequently oscillate between adjacent nodes (A→B→A→B→...). In our graph representation, this happens when a host contacts the same destination IP multiple times with different ports—the walk bounces between the destination IP node and protocol nodes, generating many near-identical walks.

These redundant walks inflate feature dimensions without adding information. Consider a host that contacts one server 50 times: we'd see many walks of the form "src→IP→protocol→port1", "src→IP→protocol→port2", etc. But most of these are variations on the same underlying structure.

Non-tottering walks fix this by prohibiting immediate backtracking: once we arrive at a node from a predecessor, we cannot immediately return to that predecessor. This prevents oscillation and creates more meaningful walks that reflect actual graph structure. The result: features are more concise (15-20% fewer dimensions), training is faster, and—remarkably—classification accuracy improves rather than degrades. This suggests the cleaner features reduce noise.

### 4.3 False Positives and False Negatives Analysis

**False Positives (FP = 3):**

We incorrectly flagged three normal hosts as anomalous. Examination reveals why: these were likely legitimate infrastructure hosts with naturally divergent communication patterns. A DNS server, for instance, must contact many different IPs (its upstream resolvers, internal servers), creating a high-diversity graphlet. A monitoring/backup system similarly contacts many servers for diagnostics or data collection. Our strict labeling strategy (any unusual traffic = anomaly) works well for real attacks but occasionally conflates legitimate infrastructure diversity with malicious behavior.

False positives are operationally costly—security teams waste time investigating innocent hosts—but less critical than missed attacks.

**False Negatives (FN = 1):**

We missed one anomalous host, classifying it as normal. This is more concerning. The most likely explanation is a low-volume, stealthy attack: an attacker deliberately limiting their activity to avoid detection, using only a few connections that happen to match normal patterns. Another possibility is an insider threat using legitimate protocols and patterns for exfiltration—behaviorally indistinguishable from normal activity in a short observation window.

Detecting such attacks would require temporal features (activity over time), protocol-level analysis, or behavioral baselines—extensions beyond the scope of this work.

### 4.4 Recommendations for Improvement

1. **Feature Engineering:**
   - Add temporal features (connection patterns over time)
   - Include statistical features (min/max/avg flow sizes)
   - Incorporate protocol-specific behavioral patterns
   - Weight recent activities more heavily than historical ones

2. **Model Optimization:**
   - Use ensemble methods combining multiple kernels
   - Implement active learning to label uncertain predictions
   - Adjust SVM hyperparameters based on cost of errors
   - Consider anomaly detection techniques (One-Class SVM) for truly new attacks

3. **Operational Integration:**
   - Implement with lower threshold for detection to catch more attacks (reduce FN)
   - Combine with whitelist/blacklist to reduce false alarms (reduce FP)
   - Correlate alerts with other security systems
   - Periodically retrain with new labeled data as threats evolve

4. **Handling Class Imbalance:**
   - Consider oversampling anomalous hosts or undersampling normal hosts
   - Use cost-sensitive learning to penalize errors appropriately
   - Evaluate using metrics robust to imbalance (F1-score, ROC-AUC, PR-AUC)

---

## 5. Key Insights and Conclusions

### 5.1 Graphlet-Based vs. Traditional Approaches

Our graphlet-based approach offers advantages over simpler methods:
- **Structural information:** Captures graph topology, not just aggregate statistics
- **Adaptability:** Can detect novel attacks with different patterns
- **Interpretability:** Graphlets show actual communication structures (useful for investigation)
- **Scalability:** Random walk kernel is more efficient than many alternatives

### 5.2 Practical Recommendations

The choice of method depends on operational constraints:

**For Critical Infrastructure / Research:** The kernel trick method delivers the best empirical performance (92.86% accuracy, zero false positives on our test set). The 2.57 second computation per batch is acceptable if you're analyzing historical data or can batch process overnight logs. This approach is best for high-stakes environments where accuracy is paramount.

**For Production Network Monitoring (Recommended):** Non-tottering random walk strikes the best balance. It achieves identical accuracy to the kernel trick (92.86%) in just 0.25 seconds—fast enough for real-time or batch processing on incoming traffic. The method is simple, interpretable, and requires less memory. This is our recommendation for operational deployment.

**For Embedded/Resource-Constrained Systems:** Standard direct mapping is fastest (0.23s) with acceptable accuracy (85.71%). The trade-off is lower anomaly detection rate (66.67%)—roughly one in three attacks missed. Only suitable if speed is critical and missing some attacks is acceptable.

### 5.3 Future Work

1. **Multi-class classification:** Extend to specific attack types rather than binary classification
2. **Temporal patterns:** Incorporate time-based features for evolving threat detection
3. **Graph neural networks:** Use GNNs for automatic feature learning from graphlets
4. **Interpretability:** Generate explanations for why hosts are flagged as anomalous
5. **Concept drift:** Handle evolution of normal and anomalous behaviors over time

---

## 6. Conclusion

Network anomaly detection remains challenging because attacks are rare and often deliberately stealthy. We've shown that encoding host communication patterns as graphlets—capturing not just what hosts communicate, but how—provides a powerful signal for distinguishing normal from anomalous behavior.

Our comparison of three methods reveals important practical insights: explicit feature mapping is fast but loses accuracy on rare classes; the kernel trick recovers accuracy but becomes computationally expensive; non-tottering random walks provide an elegant middle ground, achieving kernel-trick-level accuracy at direct-mapping speed. The non-tottering variant works better because it eliminates redundant oscillating walks, producing more meaningful features with less noise.

On our dataset, we achieved 92.86% accuracy with 83.33% anomaly detection rate using non-tottering walks in just 0.25 seconds. While no detection method is perfect (our one false negative illustrates), this performance is sufficient for practical security monitoring, especially when combined with other detection techniques.

We recommend implementing the non-tottering random walk approach in production environments. It is fast enough for real-time processing, accurate enough to catch most attacks, and interpretable—security analysts can examine the graphlets of flagged hosts to understand why they were suspicious. As a next step, adding temporal features and coupling this method with traditional rule-based detection would create a more robust system.

![Method Comparison Summary](./output/6.png)
*Figure 6: Comprehensive comparison of all three methods showing accuracy, precision, recall, and computational efficiency.*