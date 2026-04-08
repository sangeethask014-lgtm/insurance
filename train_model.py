import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4) (1).csv')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Prepare data with feature engineering
X = df[['age', 'bmi', 'children']].copy()
y = df['charges'].copy()

# Add polynomial features for better fit
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['age', 'bmi', 'children']))

print("\n\nFeature Engineering Applied:")
print(f"Original features: {X.shape[1]}")
print(f"Polynomial features (degree=2): {X_poly_df.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate multiple models with hyperparameter tuning
print("\n" + "="*80)
print("TRAINING ADVANCED MODELS WITH HYPERPARAMETER TUNING")
print("="*80)

models = {
    'Ridge (tuned)': GridSearchCV(Ridge(), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5),
    'Lasso (tuned)': GridSearchCV(Lasso(max_iter=5000), {'alpha': [0.01, 0.1, 1, 10]}, cv=5),
    'ElasticNet (tuned)': GridSearchCV(ElasticNet(max_iter=5000), {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}, cv=5),
    'Random Forest (tuned)': GridSearchCV(RandomForestRegressor(random_state=42), 
                                         {'n_estimators': [100, 200], 'max_depth': [10, 15, 20], 'min_samples_leaf': [2, 4]}, cv=5),
    'Gradient Boosting (tuned)': GridSearchCV(GradientBoostingRegressor(random_state=42),
                                             {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}, cv=5),
    'ExtraTrees (tuned)': GridSearchCV(ExtraTreesRegressor(random_state=42),
                                      {'n_estimators': [100, 200], 'max_depth': [10, 15, 20], 'min_samples_leaf': [2, 4]}, cv=5),
    'SVR (tuned)': GridSearchCV(SVR(), {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 1], 'kernel': ['rbf', 'poly']}, cv=5)
}

results = {}
best_model = None
best_rmse = float('inf')
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for linear/SVM models, original for tree-based
    if 'Random Forest' in name or 'Gradient Boosting' in name or 'ExtraTrees' in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    best_params = model.best_params_ if hasattr(model, 'best_params_') else 'N/A'
    print(f"  Best Params: {best_params}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R² Score: {r2:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name

print("\n" + "="*80)
print(f"BEST MODEL: {best_model_name}")
print(f"RMSE: {best_rmse:.2f}")
print("="*80)

# Save the best model and scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('best_model_name.pkl', 'wb') as f:
    pickle.dump(best_model_name, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save polynomial feature transformer
with open('poly_features.pkl', 'wb') as f:
    pickle.dump(poly, f)

# Save model metrics
metrics = {
    'rmse': best_rmse,
    'mae': results[best_model_name]['mae'],
    'r2': results[best_model_name]['r2']
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\nModel, scaler, and feature transformer saved successfully!")
