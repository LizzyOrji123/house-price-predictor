# 🏠 House Price Predictor

A machine learning project that uses regression techniques to predict house prices based on structured data. Built with **Python**, **Scikit-Learn**, and powered by **Kaggle’s house pricing dataset** — all developed inside **Visual Studio Code**.

---

## 🎯 Project Goal

Predict final sale prices of homes based on key features such as square footage, basement size, garage capacity, and year built. Evaluate model performance and visualize predictions versus actual data.

---

## 🛠️ Setup Instructions (VS Code + macOS/Linux)

### 🔹 Step 1: Create & Open Project Folder

```bash
mkdir ~/house-price-predictor
cd ~/house-price-predictor
code .
```

> 💡 Tip: Ensure VS Code CLI is installed. Run `Shell Command: Install 'code' command in PATH` from Command Palette (`Cmd + Shift + P`).

---

### 🔹 Step 2: Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

### 🔹 Step 3: Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

### 🔹 Step 4: Download Dataset

- Visit the Kaggle dataset [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- Download `train.csv` and move it into your project folder

---

## 📈 Model Overview

The model trains a **Linear Regression** algorithm using these features:

- `GrLivArea` (Above ground living area)
- `TotalBsmtSF` (Basement square footage)
- `OverallQual` (Overall material & finish quality)
- `GarageCars` (Garage capacity)
- `YearBuilt` (Year house was built)

It then:

- Splits data into train/test
- Fits the model
- Predicts prices
- Evaluates using **R² Score** and **RMSE**
- Visualizes results with a scatter plot

---

## 🐍 Running the Script

```bash
python main.py
```

Output includes:

```
R² Score: 0.79
RMSE: 29000.12
```

Plus a matplotlib plot showing actual vs predicted prices.

---

## 📊 Sample Plot

> ✨ Scatter plot comparing real sale prices with predicted values to assess model accuracy.

---


## 🤝 Credits

- Dataset from [Kaggle Housing Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
