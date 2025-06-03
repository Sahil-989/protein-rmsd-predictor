import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="Protein RMSD Prediction", layout="wide")


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("protein.csv")

df = load_data()

# Train model
X = df.drop(columns=['RMSD'])
y = df['RMSD']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Page title
st.title("Protein Residue Size Predictor (RMSD)")

# Sidebar inputs
st.sidebar.header("Input Feature Values")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.slider(
        f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean())
    )

# Prediction
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
st.subheader("🔍 Predicted RMSD:")
st.write(f"**{prediction:.2f}**")

# Feature Importance
st.subheader("📊 Feature Importance")
importance = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
importance.sort_values().plot(kind='barh', ax=ax, color="teal")
st.pyplot(fig)

# Correlation Matrix
st.subheader("🔗 Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)
