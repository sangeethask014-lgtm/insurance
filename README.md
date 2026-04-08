# Insurance Premium Prediction Model

A machine learning regression model that predicts insurance premiums based on age, BMI, and number of children. Built with scikit-learn and deployed with Streamlit.

## Project Overview

This project uses various regression algorithms to predict insurance charges with the optimal model achieving an **RMSE of 11,454.33**.

### Model Performance

**Best Model:** Lasso Regression (alpha=0.1)
- **RMSE:** 11,454.33
- **R² Score:** 0.1549
- **MAE:** $9,181.33

### Models Tested
- Ridge Regression (various alpha values)
- Lasso Regression (various alpha values)
- ElasticNet
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- SVR

## Project Structure

```
insurance/
├── train_model_optimized.py    # Optimized main training script
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── best_model.pkl              # Serialized trained model
├── scaler.pkl                  # Feature scaler
├── model_metrics.pkl           # Model performance metrics
└── README.md                   # This file
```

## Features

- **Age** (years): 18-65
- **BMI** (kg/m²): Body mass index
- **Children**: Number of dependents
- **Charges** (target): Insurance premium costs

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sangeethask014-lgtm/insurance.git
cd insurance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the optimized training script:
```bash
python train_model_optimized.py
```

This will:
- Load the insurance dataset
- Train multiple regression models
- Select the best performing model
- Save the model, scaler, and metrics

### Running the Streamlit App

```bash
streamlit run app.py
```

Then open your browser to:
- **Local:** http://localhost:8501
- **Network:** http://192.168.1.191:8501

## Features

### 1. Single Prediction
Input age, BMI, and number of children to get an instant premium prediction.

### 2. Batch Predictions
Upload a CSV file with multiple records to get predictions for all at once.

### 3. Data Analysis
View dataset statistics, distributions, and model performance metrics.

## Required Dependencies

- pandas==2.1.4
- numpy==1.24.3
- scikit-learn==1.3.2
- xgboost==3.2.0
- lightgbm==4.6.0
- streamlit==1.28.1

## Dataset

The model is trained on insurance premium data with 1,338 records containing:
- Age range: 18-64 years
- BMI range: 15.96-53.13 kg/m²
- Children: 0-5 dependents
- Charges range: $1,121.87-$63,770.43

## Model Insights

The analysis reveals that insurance premiums have a primarily **linear relationship** with the input features, making linear regression models (Lasso, Ridge, ElasticNet) perform better than complex tree-based models.

## Author

- **GitHub Username:** sangeethask014-lgtm
- **Email:** sangeethask014@gmail.com

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please reach out to sangeethask014@gmail.com
