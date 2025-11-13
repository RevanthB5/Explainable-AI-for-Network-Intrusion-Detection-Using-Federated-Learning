# Explainable AI for Network Intrusion Detection Using Federated Learning

This project implements a comprehensive Explainable AI (XAI) system for network intrusion detection using both **centralized** and **federated learning** approaches. The system achieves 98%+ accuracy on the CICIDS2017 dataset while providing interpretable predictions through LIME (Local Interpretable Model-agnostic Explanations) and advanced feature analysis.

### Key Highlights

- **Dual Training Paradigms**: Compare centralized vs. federated learning performance
- **Privacy-Preserving**: Federated learning achieves 98.06% accuracy while maintaining data privacy
- **Explainable AI**: LIME-based local explanations for individual predictions
- **Comprehensive Feature Analysis**: Cohen's d effect size, client variance, and combined importance scoring
- **Non-IID Data Handling**: Successfully trains across 5 clients with highly imbalanced data distributions


## Features

- Binary classification of network traffic (Benign vs. Malicious)
- Advanced feature engineering with custom composite features
- XGBoost-based feature selection (78 → 21 features)
- SMOTE-based class balancing
- Deep neural network with optimized architecture
- Federated learning with FedAvg aggregation
- LIME explainability for both training paradigms
- Comprehensive visualization suite (15+ plots)
- Feature discrimination and variance analysis


## Dataset

**CICIDS2017 Network Intrusion Dataset**

- **Source**: Canadian Institute for Cybersecurity
- **Initial Samples**: 305,185 (before preprocessing)
- **Final Samples**: 223,181 (after removing 82,004 duplicates)
- **Features**: 78 original → 21 selected features
- **Sampling**: 10% sample applied for computational efficiency
- **Attack Types**: Botnet, Brute Force, DDoS, DoS, Infiltration, Port Scan, Web Attacks

**Data Split:**

- Training: 75%
- Validation: 10%
- Test: 15%


## Model Architecture

<img width="2419" height="1479" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/37826ff7-a270-4c22-97a3-a7b8e4c2a6a9" />

**Optimizer**: Adam
**Loss Function**: Binary Cross-entropy
**Metrics**: Accuracy, Precision, Recall, F1-score

### Training Configuration

**Centralized:**

- Max Epochs: 250
- Batch Size: 1024
- Early Stopping: Patience 8
- Learning Rate Reduction: Factor 0.1, Patience 4, Min LR 1e-07

**Federated:**

- Clients: 5 (non-IID Dirichlet α=0.5)
- Rounds: 10
- Local Epochs: 5 per round
- Aggregation: FedAvg (weighted by dataset size)


## Results

### Performance Comparison

| Metric | Centralized | Federated | Difference |
| :-- | :-- | :-- | :-- |
| **Accuracy** | 98.18% | 98.06% | -0.13% |
| **Precision** | 92.13% | 92.16% | +0.03% |
| **Recall** | 96.18% | 95.23% | -0.95% |
| **F1-Score** | 94.11% | 93.67% | -0.44% |

### Confusion Matrix (Centralized)

```
                Predicted
              Benign  Malicious
Actual Benign   28,015    415
     Malicious    193   4,855
```


### Confusion Matrix (Federated)

```
                Predicted
              Benign  Malicious
Actual Benign   28,021    409
     Malicious    241   4,807
```


### Top 5 Discriminative Features (Cohen's d)

1. **PacketLengthStdDiff**: 1.580
2. **Bwd Packet Length Min**: 1.562
3. **Fwd IAT Std**: 1.516
4. **Total Fwd Packets**: 1.212
5. **Bwd Packets/s**: 1.185

### LIME Feature Importance (Centralized)

1. Fwd Packet Length Max: 0.260
2. Bwd Packet Length Min: 0.234
3. Bwd Packets/s: 0.211
4. Packet Length Std: 0.141
5. Avg Bwd Segment Size: 0.135

### LIME Feature Importance (Federated)

1. FIN Flag Count: 0.276
2. Bwd Packet Length Min: 0.266
3. Fwd Packet Length Max: 0.241
4. Bwd Packets/s: 0.205
5. Packet Length Std: 0.159

## Visualizations

**Feature Analysis**

- Feature Discrimination Power (Cohen's d)
- Feature Variance Across Clients
- Combined Feature Importance

**Training Progress**

- Validation Accuracy vs. Federated Rounds
- Validation Loss vs. Federated Rounds
- Per-Client Training Curves

**Model Performance**

- Confusion Matrices (Centralized \& Federated)
- ROC Curves
- Precision-Recall Curves

**Explainability**

- LIME Instance Explanations
- LIME Side-by-Side Comparison
- Aggregated LIME Feature Importance


## Federated Learning Details

### Client Data Distribution (Non-IID)

| Client | Samples | Benign % | Malicious % |
| :-- | :-- | :-- | :-- |
| Client 1 | 89,408 | 89.3% | 10.7% |
| Client 2 | 6,477 | 88.3% | 11.7% |
| Client 3 | 62,925 | 48.2% | 51.8% |
| Client 4 | 42,482 | 59.5% | 40.5% |
| Client 5 | 11,930 | 8.4% | 91.6% |

### Convergence Metrics

| Round | Val Accuracy | Val Loss |
| :-- | :-- | :-- |
| 1 | 96.38% | 0.1693 |
| 5 | 97.82% | 0.0712 |
| 10 | 98.20% | 0.0547 |

## Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Class Balancing**: imbalanced-learn (SMOTE)
- **Explainability**: LIME
- **Visualization**: matplotlib, seaborn
- **Federated Learning**: Custom FedAvg implementation


## Key Findings

1. **Minimal Performance Gap**: Federated learning achieves within 0.13% accuracy of centralized approach while preserving privacy
2. **Packet Length Variability**: Strongest indicator of malicious traffic (Cohen's d = 1.580)
3. **Protocol Features**: Federated model emphasizes TCP flags (FIN) more than centralized
4. **Non-IID Robustness**: Successfully handles extreme data imbalance (8.4% - 91.6% malicious ratio)
5. **Convergence**: Both models achieve optimal performance in 30-40 epochs

## Future Work

- Extend to multi-class classification (8 attack types)
- Implement differential privacy mechanisms
- Integrate SHAP for global feature importance
- Test on additional datasets (NSL-KDD, UNSW-NB15)
- Explore transformer-based architectures
- Implement asynchronous federated learning
- Deploy real-time inference pipeline
- Create interactive explainability dashboard
