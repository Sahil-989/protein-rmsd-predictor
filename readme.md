# 🧬 Protein RMSD Prediction App

This is an interactive machine learning app built using **Streamlit** that predicts **Root Mean Square Deviation (RMSD)** of proteins based on structural and energy features.

## 🎯 Project Overview

- Built with `RandomForestRegressor` from scikit-learn
- Real-time predictions using feature sliders
- Feature importance visualization
- Correlation heatmap of the dataset
- Streamlit-based front-end interface

## 📊 Dataset

The dataset (`protein.csv`) includes computed features (f1 to f9, uG, uA) and the target variable `RMSD`. These represent molecular geometry, energy, and structural parameters derived from protein models.

## 📷 Screenshots

<img src="screenshot.png" width="600"/>

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/protein-rmsd-predictor.git
cd protein-rmsd-predictor
