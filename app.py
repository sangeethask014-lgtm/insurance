import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set page config
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("🔮 Insurance Premium Prediction Model")
st.markdown("---")

# Load or train model
@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    
    # Check if model files exist
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
        try:
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('model_metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            return model, scaler, metrics, "Loaded"
        except:
            pass
    
    # Train model if files don't exist
    st.info("🔨 Training model on first load...")
    
    try:
        # Load and prepare data
        df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4) (1).csv')
        X = df[['age', 'bmi', 'children']]
        y = df['charges']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Lasso model
        model = Lasso(alpha=0.1, max_iter=5000)
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        # Save files
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('model_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        
        return model, scaler, metrics, "Trained"
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, "Error"

model, scaler, metrics, status = load_or_train_model()

if model is None:
    st.error("Failed to load or train model. Please check the data file.")
    st.stop()

# Display model info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", "Lasso Regression")
with col2:
    if metrics:
        st.metric("RMSE", f"${metrics['rmse']:,.0f}")
    else:
        st.metric("RMSE", "N/A")
with col3:
    if metrics:
        st.metric("R² Score", f"{metrics['r2']:.4f}")
    else:
        st.metric("R² Score", "N/A")
with col4:
    st.metric("Status", f"✅ {status}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Data Analysis"])

# Tab 1: Single Prediction
with tab1:
    st.header("Make a Single Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)", min_value=18, max_value=65, value=30)
    
    with col2:
        bmi = st.slider("BMI (kg/m²)", min_value=15.0, max_value=55.0, value=25.0, step=0.1)
    
    with col3:
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
    
    if st.button("📈 Predict Premium", key="single", use_container_width=True):
        # Prepare input
        input_data = np.array([[age, bmi, children]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"### Predicted Premium: ${prediction:,.2f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Age:** {age} years")
        with col2:
            st.info(f"**BMI:** {bmi} kg/m²")
        with col3:
            st.info(f"**Children:** {children}")

# Tab 2: Batch Prediction
with tab2:
    st.header("Upload CSV for Batch Predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['age', 'bmi', 'children']
            if all(col in df.columns for col in required_cols):
                st.success(f"✅ File loaded! {len(df)} records found.")
                
                X = df[required_cols]
                
                # Apply scale and predict
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                
                # Add predictions to dataframe
                df['Predicted_Premium'] = predictions
                
                # Display results
                st.dataframe(df, use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Tab 3: Data Analysis
with tab3:
    st.header("Dataset Analysis")
    
    # Load original data
    try:
        df = pd.read_csv('insurance_premium_correct - insurance_premium (2) (2) (4) (1).csv')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Avg Age", f"{df['age'].mean():.1f} years")
        with col3:
            st.metric("Avg BMI", f"{df['bmi'].mean():.1f} kg/m²")
        with col4:
            st.metric("Avg Premium", f"${df['charges'].mean():,.2f}")
        
        st.markdown("---")
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            st.bar_chart(df['age'].value_counts().sort_index())
        
        with col2:
            st.subheader("Premium Distribution")
            st.bar_chart(df['charges'].value_counts().sort_index())
        
        st.markdown("---")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Insurance Premium Prediction Model | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
