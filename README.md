# Exoplanet Classification and Analysis Project

A comprehensive machine learning project for analyzing and classifying exoplanets using the Planetary Habitability Laboratory (PHL) Exoplanet Catalog dataset.

## 🚀 Project Overview

This project implements a complete machine learning pipeline to analyze exoplanet data and predict habitability characteristics. The analysis includes data preprocessing, exploratory data analysis, feature engineering, model training with multiple algorithms, and model explainability using LIME (Local Interpretable Model-agnostic Explanations).

## 📊 Dataset

The project uses the **PHL Exoplanet Catalog** dataset containing:
- **5,600+ exoplanets** with comprehensive astronomical measurements
- **100+ features** including planetary and stellar characteristics
- Physical properties: mass, radius, orbital period, temperature
- Stellar properties: luminosity, metallicity, habitable zone boundaries
- Habitability indicators: ESI (Earth Similarity Index), habitable zone classification

### Key Features Analyzed:
- **Planetary Properties**: Mass, radius, orbital characteristics, temperature
- **Stellar Properties**: Temperature, mass, luminosity, metallicity
- **Habitability Metrics**: Habitable zone classification, ESI scores
- **Detection Methods**: Transit, radial velocity, microlensing, and others

## 🛠️ Technologies & Libraries

### Core Libraries:
- **Data Processing**: `pandas`, `numpy`, `dask`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Explainable AI**: `LIME`
- **Imbalanced Learning**: `imbalanced-learn` (SMOTE)

### Machine Learning Models:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

## 📁 Project Structure

```
EXOplanet/
├── README.md                                    # Project documentation
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore file
├── sec1_group11_24341216_20201010.ipynb        # Main analysis notebook
├── exoplanet_dataset_phl.csv                   # Raw dataset
├── EXOplanet_report.pdf                        # Analysis report
└── docs/                                        # Additional documentation
    └── methodology.md                           # Detailed methodology
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Analysis**: Comprehensive analysis of data completeness
- **Error Feature Removal**: Elimination of uncertainty/error columns
- **Feature Selection**: Removal of redundant identifiers and low-quality features
- **Data Cleaning**: Handling of categorical and numerical inconsistencies

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries and distributions
- Correlation analysis between features
- Visualization of key relationships
- Class imbalance assessment

### 3. Feature Engineering
- **Imputation Strategies**: KNN and Iterative imputation for missing values
- **Scaling**: StandardScaler and MinMaxScaler normalization
- **Encoding**: Label encoding for categorical variables
- **Resampling**: SMOTE for handling class imbalance

### 4. Model Training & Evaluation
- **Cross-validation**: Stratified k-fold validation
- **Hyperparameter Tuning**: RandomizedSearchCV optimization
- **Performance Metrics**: 
  - Classification accuracy
  - ROC-AUC scores
  - F-beta scores
  - Confusion matrices

### 5. Model Explainability
- **LIME Analysis**: Local explanations for individual predictions
- **Feature Importance**: Permutation importance analysis
- **Partial Dependence**: Understanding feature-target relationships

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/EXOplanet.git
   cd EXOplanet
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

5. **Open the main analysis notebook**:
   - Navigate to `sec1_group11_24341216_20201010.ipynb`
   - Execute cells sequentially

## 📈 Key Results

### Model Performance Summary:
- **Best Performing Model**: [To be determined after analysis]
- **Accuracy**: [Results from notebook execution]
- **ROC-AUC**: [Results from notebook execution]
- **Key Features**: [Important features identified]

### Insights Discovered:
- Correlation patterns between planetary and stellar characteristics
- Key indicators of exoplanet habitability
- Feature importance rankings for classification tasks
- Class distribution and imbalance patterns

## 🔍 Usage Examples

### Basic Analysis:
```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('exoplanet_dataset_phl.csv')

# Basic exploration
print(f"Dataset shape: {data.shape}")
print(f"Missing values: {data.isnull().sum().sum()}")
```

### Model Training:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data (after preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Distribution plots** for numerical features
- **Correlation heatmaps** for feature relationships
- **Class distribution** charts
- **Model performance** comparison plots
- **Feature importance** rankings
- **LIME explanations** for individual predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References & Data Sources

- **Planetary Habitability Laboratory (PHL)**: [University of Puerto Rico at Arecibo](http://phl.upr.edu/)
- **Exoplanet Catalog**: PHL's Habitable Exoplanets Catalog
- **Detection Methods**: Various space missions and ground-based observatories
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost documentation

## 📧 Contact

For questions or collaborations, please reach out through:
- GitHub Issues for technical questions
- Email: [Your email here]
- LinkedIn: [Your LinkedIn profile]

## 🙏 Acknowledgments

- Planetary Habitability Laboratory for providing the comprehensive exoplanet dataset
- The astronomical community for continuous exoplanet discovery efforts
- Open-source machine learning community for excellent tools and libraries
- Contributors and reviewers who helped improve this project

---

**Note**: This project is for educational and research purposes. Results should be validated with domain experts before any scientific conclusions.
