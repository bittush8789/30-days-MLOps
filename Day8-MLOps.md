# 🚀 Day 8 of 30 Days of MLOps Challenge: Model Evaluation & Metrics

## 📚 Key Learnings
- Importance of evaluation metrics in ML.
- Core classification metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- Difference between classification and regression metrics.
- Understanding Precision-Recall tradeoff.
- Visual tools: ROC curve, PR curve, Confusion Matrix heatmap.
- Importance of evaluation consistency during training and deployment.

## 🧠 ML Evaluation Metrics
Machine learning evaluation metrics assess how well a model performs. Metrics differ based on the ML task.

---

## **1. Classification Metrics**
| Metric | Description |
|--------|-------------|
| Accuracy | Correct predictions / Total predictions |
| Precision | True Positives / Predicted Positives |
| Recall | True Positives / Actual Positives |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Ability to separate classes |
| Confusion Matrix | TP, FP, FN, TN breakdown |

---

## **2. Regression Metrics**
| Metric | Description |
|--------|-------------|
| MAE | Avg absolute error |
| MSE | Avg squared error (penalizes large errors) |
| RMSE | sqrt(MSE) |
| R² Score | Variance explained (1 is perfect) |

---

## **3. Clustering Metrics**
| Metric | Description |
|--------|-------------|
| Silhouette Score | Similarity within vs outside cluster |
| ARI | Measures clustering similarity |
| DB Index | Lower is better |

---

## 🔥 Why Metrics Matter  
Evaluation metrics ensure:
- Model selection
- Hyperparameter tuning
- Monitoring in production
- Bias & fairness detection
- Business value measurement

---

## ⚠️ Example: Fraud Detection
Accuracy may show **98%** but still miss fraud cases.  
Better metrics → Precision, Recall, F1, ROC-AUC.

---

## 🆚 Classification vs Regression (Quick Reference)
| Feature | Classification | Regression |
|---------|---------------|------------|
| Output | Class labels | Continuous values |
| Examples | Spam detection | Price prediction |
| Metrics | Accuracy, F1, ROC-AUC | MAE, RMSE, R² |

---

## 🎯 Precision-Recall Tradeoff
- High Precision → fewer false positives  
- High Recall → fewer false negatives  
- Tradeoff controlled using threshold  
- Use case dependent (spam → Precision, cancer detection → Recall)

---

## 📊 Visual Model Evaluation Tools
### **1. ROC Curve**
- Plots TPR vs FPR  
- Good for balanced datasets  

### **2. Precision-Recall Curve**
- Better for imbalanced datasets  

### **3. Confusion Matrix Heatmap**
- Shows TP/FP/TN/FN visually  
- Useful for error analysis  

---

## 🔄 Importance of Evaluation Consistency
- Ensures reliability between training & production
- Helps debugging and monitoring
- Supports compliance and reproducibility
- Prevents metric mismatch issues

---

# 🧪 Hands-on Example (Titanic Dataset)

### **1. Install Required Libraries**
```
pip install scikit-learn matplotlib seaborn pandas
```

### **2. Load & Preprocess**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('train.csv')
X = df[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(df.mean())
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### **3. Train Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

### **4. Evaluate Model**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
```

### **5. Visualizations**
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve')
plt.show()
```

---

# 🔥 Challenges
- Train a binary classifier and evaluate using 5+ metrics  
- Plot confusion matrix  
- Plot ROC curve and compute AUC  
- Compare 2 models  
- Handle imbalanced dataset  
- Save metrics in JSON/CSV  
- Log results to MLflow/W&B  

---

## 🤷🏻 How to Participate?
- Complete tasks  
- Document progress on GitHub/Medium/Hashnode  

**Keep Learning… 🚀**