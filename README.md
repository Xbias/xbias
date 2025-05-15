# X-Bias

**X-Bias** is a fairness evaluation tool designed to reveal **attributional bias** in machine learning models. Traditional fairness metrics assess disparities in prediction outcomes, but X-Bias focuses on *how* models make decisions by comparing **feature attribution patterns** across demographic groups. This helps detect hidden biases in the reasoning process of AI systems.

---

## 🔍 Why X-Bias?

In many high-stakes domains—such as credit scoring, hiring, and healthcare—models can satisfy conventional fairness metrics while relying on different features for different groups. X-Bias uncovers this type of bias by analyzing **local explanation vectors**, making it a powerful complement to output-based fairness assessments.

---

## ✨ Features

- Computes disparities in feature attributions across groups
- Works with multiple model types: logistic regression, EBMs, and MLPs
- Supports explanation tools like `interpret` and `SHAP`
- Compatible with common datasets (e.g., Adult, German Credit, Taiwan Credit)
- Multiple distance metrics: L1, L2, and Cosine distance
- Easy integration into fairness audits and ML pipelines

---

## ⚙️ Installation

Install using `pip`:

```bash
pip install -r requirements.txt
```
## 📈 How It Works
Train a model on your dataset.

Generate local explanations for each instance (e.g., feature attributions).

Group instances based on a sensitive attribute (e.g., gender, race).

Compute the average explanation vector for each group.

Measure the divergence between the groups using a distance metric (L1, L2, or cosine).

## 🧪 Example Usage
```python

from xbias import compute_xbias
from sklearn.linear_model import LogisticRegression
import numpy as np

# Assume you have X_train, y_train, X_test, sensitive_attr defined
model = LogisticRegression()
model.fit(X_train, y_train)

# Get local explanations (for linear models, use coefficients)
explanations = X_test * model.coef_

# Split explanation vectors by sensitive attribute
group_0 = explanations[sensitive_attr == 0]
group_1 = explanations[sensitive_attr == 1]

# Compute X-Bias (L1 distance)
xbias_score = compute_xbias(group_0, group_1, metric="l1")
print("X-Bias (L1):", xbias_score)

```
## 📊 Supported Distance Metrics
Cosine Distance
Measures the angle (dissimilarity) between group attribution vectors.

## 📂 Example Datasets
You can evaluate X-Bias on public datasets commonly used in fairness research:

UCI Adult Dataset

German Credit Dataset

Taiwan Credit Dataset

Dataset loaders are provided in the data/ folder.

## 📁 Project Structure
```bash

xbias/
├── data/               # Dataset loaders
├── explainers/         # Feature attribution tools
├── metrics/            # X-Bias computation functions
├── utils/              # Helper functions
├── notebooks/          # Demo notebooks and experiments
├── requirements.txt
└── README.md
```
## 📚 Citation
If you use X-Bias in your work, please cite:

```bibtex
@article{shoeibi2024measuring,
  title={Measuring Attributional Bias in AI Explanations: A New Lens on Fairness},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```
## 🤝 Contributing
We welcome contributions! Feel free to open an issue or submit a pull request for improvements, features, or bug fixes.

## 📜 License
