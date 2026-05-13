# 🏥 Health Insurance Premium Predictor

A machine learning project that predicts annual health insurance charges based on patient profile — built using Python, Scikit-learn, and Streamlit.

---

## 🎯 Project Overview

Health insurance pricing is one of the most critical challenges in the healthcare industry. This project builds and compares multiple ML models to predict insurance premiums based on demographic and health factors, providing actionable insights into what drives costs.

**Why this matters:** Companies like health insurers and healthtech platforms use models like these to personalize pricing, detect anomalies, and serve customers better.

---

## 📊 Dataset

- **Source:** [Medical Cost Personal Dataset — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size:** 1,338 records, 7 features
- **Target:** `charges` — annual insurance cost in USD

| Feature | Description |
|---|---|
| age | Age of the insured |
| sex | Male / Female |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | US region (NE/NW/SE/SW) |
| charges | Annual insurance charges (target) |

---

## 🔍 Key Findings

- **Smoking** is the #1 cost driver — smokers pay **3.8x more** than non-smokers
- **Age** and **BMI** are the next strongest predictors
- **Gradient Boosting** achieved the best R² score of **~0.87**

---

## 🤖 Models Trained & Compared

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | ~0.75 | ~$4,200 |
| Random Forest | ~0.86 | ~$2,600 |
| Gradient Boosting | ~0.87 | ~$2,400 |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/health-insurance-predictor.git
cd health-insurance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Download `insurance.csv` and place it in the project folder

### 4. Run the ML script
```bash
python health_insurance_predictor.py
```

### 5. Launch the Streamlit web app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
health-insurance-predictor/
│
├── health_insurance_predictor.py  # Full ML pipeline (EDA + models + evaluation)
├── app.py                          # Interactive Streamlit web app
├── insurance.csv                   # Dataset (download from Kaggle)
├── requirements.txt                # Dependencies
├── eda_analysis.png                # EDA visualizations
├── model_comparison.png            # Model comparison chart
├── feature_importance.png          # Feature importance chart
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Scikit-learn** — ML models, preprocessing, evaluation
- **Matplotlib / Seaborn** — visualizations
- **Streamlit** — interactive web app

---

## 💡 What I Learned

- End-to-end ML pipeline: EDA → preprocessing → model training → evaluation
- Comparing multiple regression models using R², MAE, RMSE
- Feature importance analysis to extract business insights
- Deploying an ML model as an interactive web app using Streamlit
- How smoking and BMI create non-linear effects on insurance costs

---

## 📬 Contact

Gowtham K
- LinkedIn: www.linkedin.com/in/gowtham-k-5807971b3
- Email: kgowtham590@gmail.com
