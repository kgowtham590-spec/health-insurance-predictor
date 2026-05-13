# ============================================================
# Health Insurance Premium Predictor
# Author: [Your Name]
# Description: Predicts health insurance charges using ML models
# Dataset: Medical Cost Personal Dataset (Kaggle)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
# Download from: https://www.kaggle.com/datasets/mirichoi0218/insurance
# Save as 'insurance.csv' in the same folder as this script

df = pd.read_csv('insurance.csv')

print("=" * 50)
print("STEP 1: Dataset Overview")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn Info:")
print(df.info())
print(f"\nBasic Statistics:\n{df.describe()}")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Exploratory Data Analysis")
print("=" * 50)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Distribution of target variable
print(f"\nInsurance Charges Stats:")
print(f"  Min:    ₹{df['charges'].min():,.2f}")
print(f"  Max:    ₹{df['charges'].max():,.2f}")
print(f"  Mean:   ₹{df['charges'].mean():,.2f}")
print(f"  Median: ₹{df['charges'].median():,.2f}")

# Key insight: Smoking vs Non-smoking
smoker_avg = df[df['smoker'] == 'yes']['charges'].mean()
non_smoker_avg = df[df['smoker'] == 'no']['charges'].mean()
print(f"\nKey Insight - Smoking Impact:")
print(f"  Smoker avg charges:     ₹{smoker_avg:,.2f}")
print(f"  Non-smoker avg charges: ₹{non_smoker_avg:,.2f}")
print(f"  Smokers pay {smoker_avg/non_smoker_avg:.1f}x more!")

# Plot EDA charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Health Insurance Data - Exploratory Analysis', fontsize=16, fontweight='bold')

# 1. Charges distribution
axes[0, 0].hist(df['charges'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 0].set_title('Distribution of Insurance Charges')
axes[0, 0].set_xlabel('Charges ($)')
axes[0, 0].set_ylabel('Frequency')

# 2. Smoker vs Non-smoker charges
smoker_data = [df[df['smoker'] == 'yes']['charges'], df[df['smoker'] == 'no']['charges']]
axes[0, 1].boxplot(smoker_data, labels=['Smoker', 'Non-Smoker'], patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', color='red'))
axes[0, 1].set_title('Charges: Smoker vs Non-Smoker')
axes[0, 1].set_ylabel('Charges ($)')

# 3. Age vs Charges
scatter = axes[1, 0].scatter(df['age'], df['charges'],
                              c=df['smoker'].map({'yes': 'red', 'no': 'steelblue'}),
                              alpha=0.5, s=30)
axes[1, 0].set_title('Age vs Charges (Red=Smoker, Blue=Non-Smoker)')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Charges ($)')

# 4. BMI vs Charges
axes[1, 1].scatter(df['bmi'], df['charges'],
                   c=df['smoker'].map({'yes': 'red', 'no': 'steelblue'}),
                   alpha=0.5, s=30)
axes[1, 1].set_title('BMI vs Charges (Red=Smoker, Blue=Non-Smoker)')
axes[1, 1].set_xlabel('BMI')
axes[1, 1].set_ylabel('Charges ($)')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA chart saved as 'eda_analysis.png'")

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: Data Preprocessing")
print("=" * 50)

df_processed = df.copy()

# Encode categorical variables
le = LabelEncoder()
df_processed['sex'] = le.fit_transform(df_processed['sex'])          # male=1, female=0
df_processed['smoker'] = le.fit_transform(df_processed['smoker'])    # yes=1, no=0
df_processed['region'] = le.fit_transform(df_processed['region'])    # encoded 0-3

print("Categorical columns encoded successfully.")
print(f"\nProcessed dataset preview:\n{df_processed.head()}")

# Features and Target
X = df_processed.drop('charges', axis=1)
y = df_processed['charges']

print(f"\nFeatures (X): {list(X.columns)}")
print(f"Target (y): charges")

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# Feature Scaling (for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 4: MODEL TRAINING & COMPARISON
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: Training & Comparing 3 Models")
print("=" * 50)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'model': model, 'predictions': y_pred}

    print(f"\n📊 {name}:")
    print(f"   R² Score : {r2:.4f}  ({r2*100:.1f}% variance explained)")
    print(f"   MAE      : ${mae:,.2f}  (avg prediction error)")
    print(f"   RMSE     : ${rmse:,.2f}")

# ============================================================
# STEP 5: MODEL COMPARISON CHART
# ============================================================
model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
mae_scores = [results[m]['MAE'] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Comparison', fontsize=15, fontweight='bold')

colors = ['#4C72B0', '#55A868', '#C44E52']

bars1 = axes[0].bar(model_names, r2_scores, color=colors, edgecolor='white', linewidth=1.2)
axes[0].set_title('R² Score (Higher = Better)')
axes[0].set_ylabel('R² Score')
axes[0].set_ylim(0, 1)
for bar, score in zip(bars1, r2_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

bars2 = axes[1].bar(model_names, mae_scores, color=colors, edgecolor='white', linewidth=1.2)
axes[1].set_title('Mean Absolute Error (Lower = Better)')
axes[1].set_ylabel('MAE ($)')
for bar, score in zip(bars2, mae_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'${score:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model comparison chart saved as 'model_comparison.png'")

# ============================================================
# STEP 6: BEST MODEL - FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: Feature Importance (Random Forest)")
print("=" * 50)

best_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'],
         color='steelblue', edgecolor='white')
plt.title('Feature Importance - What Drives Insurance Costs?', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Feature importance chart saved as 'feature_importance.png'")

# ============================================================
# STEP 7: PREDICT FOR A NEW PERSON
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: Sample Prediction")
print("=" * 50)

# Sample: 35-year-old male, BMI 28, 2 children, non-smoker, southeast
sample = pd.DataFrame({
    'age': [35],
    'sex': [1],        # male
    'bmi': [28.0],
    'children': [2],
    'smoker': [0],     # non-smoker
    'region': [2]      # southeast
})

predicted_charge = best_model.predict(sample)[0]
print(f"Sample Profile: 35yr male, BMI 28, 2 kids, Non-Smoker")
print(f"Predicted Insurance Premium: ${predicted_charge:,.2f}/year")

# Smoker version
sample_smoker = sample.copy()
sample_smoker['smoker'] = 1
predicted_smoker = best_model.predict(sample_smoker)[0]
print(f"\nSame profile but SMOKER: ${predicted_smoker:,.2f}/year")
print(f"Smoking costs extra: ${predicted_smoker - predicted_charge:,.2f}/year")

print("\n" + "=" * 50)
print("✅ PROJECT COMPLETE!")
print("=" * 50)
print("""
SUMMARY:
- Loaded & explored 1,338 insurance records
- Performed EDA with 4 visualizations
- Preprocessed & encoded categorical features
- Trained 3 ML models: Linear Regression, Random Forest, Gradient Boosting
- Compared models using R², MAE, RMSE
- Identified top factors driving insurance costs
- Made real-world predictions

FILES GENERATED:
  - eda_analysis.png
  - model_comparison.png
  - feature_importance.png
""")
