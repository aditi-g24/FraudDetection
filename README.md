# Fraud Detection System

A machine learning-based fraud detection system that analyzes financial transactions to identify fraudulent activities using Logistic Regression and Random Forest algorithms.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a fraud detection system using machine learning techniques to identify fraudulent transactions in financial data. The system compares two classification algorithms:
- **Logistic Regression**
- **Random Forest Classifier**

## Features

- **Data Preprocessing**: Handles missing values, duplicates, and feature engineering
- **Exploratory Data Analysis**: Comprehensive visualizations and statistical analysis
- **Multiple ML Models**: Comparison between Logistic Regression and Random Forest
- **Feature Importance Analysis**: Identifies key factors in fraud detection
- **Performance Metrics**: Accuracy, ROC-AUC, Confusion Matrix, Classification Report
- **Model Persistence**: Save and load trained models
- **Visualization**: ROC curves, confusion matrices, and distribution plots

## Dataset

The dataset contains financial transaction records with the following features:
- `step`: Time step of the transaction
- `type`: Type of transaction (PAYMENT, TRANSFER, CASH_OUT, etc.)
- `amount`: Transaction amount
- `nameOrig`: Origin account name
- `oldbalanceOrg`: Original balance before transaction
- `newbalanceOrig`: New balance after transaction
- `nameDest`: Destination account name
- `oldbalanceDest`: Destination original balance
- `newbalanceDest`: Destination new balance
- `isFraud`: Target variable (1 = Fraud, 0 = Legitimate)

**Dataset Statistics**:
- Total Records: ~14,000+
- Features: 11
- Classes: Binary (Fraud/Not Fraud)

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. Place your dataset (`Fraud.csv`) in the `Downloads` folder or update the path in the script

2. Run the main script:
```bash
python fraud_detection.py
```

### Using Jupyter Notebook

```bash
jupyter notebook fraud_detection.ipynb
```

### Loading Saved Models

```python
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
predictions = model.predict(new_data)
```

## Model Performance

### Logistic Regression
- **Accuracy**: ~XX.XX%
- **ROC-AUC Score**: ~X.XXX

### Random Forest
- **Accuracy**: ~XX.XX%
- **ROC-AUC Score**: ~X.XXX

*(Update with actual results after running)*

## Project Structure

```
fraud-detection/
│
├── fraud_detection.py          # Main Python script
├── fraud_detection.ipynb       # Jupyter notebook version
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # License file
│
├── data/
│   └── Fraud.csv              # Dataset (add to .gitignore)
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── roc_curve_comparison.png
│   └── confusion_matrices.png
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations
- **pickle** - Model serialization

## Results

### Key Findings

1. **Feature Importance**: Transaction amount and balance changes are strong indicators of fraud
2. **Model Comparison**: Random Forest outperforms Logistic Regression in detecting fraud
3. **Class Imbalance**: Dataset shows significant imbalance between fraud and legitimate transactions

### Visualizations

The project generates several visualizations:
- Correlation heatmap showing relationships between features
- ROC curves comparing model performance
- Feature importance rankings
- Confusion matrices for both models
- Distribution plots for transaction amounts

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## Acknowledgments

- Dataset source: [Specify source if applicable]
- Inspired by real-world fraud detection systems
- Built as part of [course/project name]

## Contact

For questions or feedback, please reach out to: your.email@example.com

---

⭐ If you found this project helpful, please give it a star!
