import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4) (1).csv')

print("Dataset shape:", df.shape)

# Prepare data
X = df[['age', 'bmi', 'children']].copy()
y = df['charges'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("TRAINING OPTIMIZED MODELS")
print("="*80)

# Advanced models with scikit-learn only (cloud compatible)
models = {
    'Ridge (alpha=1)': Ridge(alpha=1),
    'Ridge (alpha=10)': Ridge(alpha=10),
    'Ridge (alpha=50)': Ridge(alpha=50),
    'Ridge (alpha=100)': Ridge(alpha=100),
    'Lasso (alpha=0.01)': Lasso(alpha=0.01, max_iter=5000),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, max_iter=5000),
    'Lasso (alpha=1)': Lasso(alpha=1, max_iter=5000),
    'ElasticNet (alpha=0.1, l1=0.2)': ElasticNet(alpha=0.1, l1_ratio=0.2, max_iter=5000),
    'ElasticNet (alpha=0.1, l1=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    'Random Forest (100 trees)': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Random Forest (200 trees)': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=3, random_state=42),
}

results = {}
best_model = None
best_rmse = float('inf')
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for linear models, original for tree-based
    if any(x in name for x in ['Ridge', 'Lasso', 'ElasticNet']):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
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

# Save model metrics
metrics = {
    'rmse': best_rmse,
    'mae': results[best_model_name]['mae'],
    'r2': results[best_model_name]['r2']
}

with open('model_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\nModel and scaler saved successfully!")
