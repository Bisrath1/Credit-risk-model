# Credit-risk-model
## Credit Scoring Business Understanding

This section outlines the business context and rationale for developing a credit scoring model for Bati Bank’s buy-now-pay-later service, addressing key considerations from the Basel II Capital Accord and the project’s data constraints.

1. **Basel II Accord and Model Interpretability**  
   The Basel II Accord emphasizes accurate credit risk measurement to ensure financial stability, requiring banks to maintain sufficient capital reserves. Interpretable and well-documented models are critical to comply with Basel II’s supervisory review and market discipline pillars. For Bati Bank, an interpretable model ensures transparency in how risk scores are derived, fostering trust in the buy-now-pay-later service. Thorough documentation supports regulatory audits, enabling validation of the model’s alignment with lending decisions.

2. **Necessity and Risks of a Proxy Variable**  
   Lacking a direct "default" label, a proxy variable derived from behavioral data (e.g., RFM patterns) is essential to estimate credit risk and categorize customers as high or low risk. This enables Bati Bank to make informed loan approval decisions. However, reliance on a proxy introduces risks: if the proxy poorly correlates with actual default behavior, misclassifications could occur. Overestimating risk may exclude creditworthy customers, reducing revenue, while underestimating risk could increase defaults, leading to financial losses and regulatory challenges.

3. **Trade-offs Between Simple and Complex Models**  
   In a regulated financial context, simple models like Logistic Regression with Weight of Evidence offer interpretability, aligning with Basel II’s transparency requirements, but may lack accuracy for complex datasets. Complex models like Gradient Boosting provide higher predictive performance by capturing non-linear patterns but are less interpretable, complicating regulatory compliance. For Bati Bank, a simple model prioritizes regulatory alignment, while a complex model could enhance accuracy if paired with interpretability tools like SHAP values to meet compliance needs.


# Credit Risk Probability Model for Alternative Data

## Project Overview

This repository contains the implementation of a **Credit Risk Probability Model** developed for the 10 Academy Artificial Intelligence Mastery program (June 25 - July 1, 2025). The project, undertaken for Bati Bank in partnership with an eCommerce platform, enables a buy-now-pay-later (BNPL) service by assessing customer creditworthiness using transactional data from the Xente Challenge dataset (available on [Kaggle](https://www.kaggle.com/competitions/xente-fraud-detection/data)).

### Business Objective

Bati Bank aims to provide credit for online purchases, minimizing default risk while promoting financial inclusion. The model leverages alternative data (Recency, Frequency, Monetary - RFM metrics) to:
- Define a proxy variable for credit risk.
- Select predictive features for default likelihood.
- Assign risk probabilities and credit scores.
- Support optimal loan amount and duration predictions.

The project aligns with Basel II Capital Accord requirements, emphasizing interpretable and auditable risk measurement for regulatory compliance and informed lending decisions.

### Key Features

- **Data Processing**: Automated feature engineering pipeline using `sklearn.pipeline.Pipeline` for RFM metrics, temporal features, and categorical encoding.
- **Model Development**: Trained and tuned Logistic Regression and Random Forest models, tracked with MLflow, achieving ROC-AUC > 0.85.
- **Deployment**: Containerized FastAPI application with a `/predict` endpoint for real-time risk probability predictions.
- **CI/CD**: Automated testing and linting via GitHub Actions, ensuring code quality and reliability.
- **MLOps**: Experiment tracking and model versioning with MLflow, enabling reproducibility and scalability.

## Project Structure

The repository follows a standardized structure for clarity and maintainability:

```plaintext
credit-risk-model/
├── .github/workflows/ci.yml           # GitHub Actions CI/CD pipeline
├── data/                              # Data directory (ignored in .gitignore)
│   ├── raw/                          # Raw Xente dataset
│   └── processed/                    # Processed data (e.g., RFM features, is_high_risk)
├── notebooks/
│   └── 1.0-eda.ipynb                 # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # Feature engineering pipeline
│   ├── train.py                      # Model training and MLflow tracking
│   ├── predict.py                    # Inference script
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── pydantic_models.py        # Pydantic models for API validation
├── tests/
│   └── test_data_processing.py        # Unit tests for data processing
├── Dockerfile                        # Docker configuration for API
├── docker-compose.yml                # Docker Compose for API and MLflow
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files/directories to ignore
├── final_submission_report.md        # Final project report
└── README.md                         # This file
```



### Credit Risk Model for Bati Bank

#### Setup Instructions
- **Prerequisites**: Python 3.9, Docker, Git
- **Installation**:
  ```bash
  git clone <https://github.com/Bisrath1/Credit-risk-model.git>
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

#### Key Features
- 🏗️ RFM-based feature engineering
- 🤖 MLflow-tracked model training (Logistic Regression + Random Forest)
- 🚀 FastAPI deployment with Docker
- ✅ CI/CD with GitHub Actions

#### Business Value
- 📈 Enables BNPL services for underserved customers
- ⚖️ Basel II compliant risk scoring
- 💰 ROC-AUC > 0.85 prediction accuracy

#### Quick Start
1. Process data:
   ```bash
   python src/data_processing.py
   ```
2. Train models:
   ```bash
   python src/train.py
   ```
3. Deploy API:
   ```bash
   docker-compose up -d
   ```

#### Future Roadmap
- 🌐 Cloud deployment (AWS)
- 🔄 Real-time monitoring
- 📱 Alternative data integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```
