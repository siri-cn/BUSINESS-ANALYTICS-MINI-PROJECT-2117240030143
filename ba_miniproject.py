import os
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------------------------------
# 1. LOAD / SIMULATE DATASET (Kaggle Schema: sowmihari/returns-management)
# -------------------------------------------------------------------------
dataset_path = "returns_sustainability_dataset.csv"

if os.path.exists(dataset_path):
    print(f"[*] Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
else:
    print(f"[*] '{dataset_path}' not found. Initializing dataset with Kaggle schema...")
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Order_ID': [f'ORD_{i:05d}' for i in range(1, n + 1)],
        'Product_ID': [f'PROD_{np.random.randint(100, 500)}' for _ in range(n)],
        'User_ID': [f'USR_{np.random.randint(1000, 3000)}' for _ in range(n)],
        'Order_Date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'Product_Category': np.random.choice(['Clothing', 'Electronics', 'Books', 'Toys', 'Home & Kitchen'], size=n, p=[0.35, 0.25, 0.15, 0.1, 0.15]),
        'Product_Price': np.random.exponential(scale=60, size=n) + 15,
        'Order_Quantity': np.random.choice([1, 2, 3, 4], size=n, p=[0.65, 0.2, 0.1, 0.05]),
        'Discount_Applied': np.random.choice([0.0, 0.1, 0.2, 0.3, 0.5], size=n),
        'User_Age': np.random.randint(18, 68, size=n),
        'User_Gender': np.random.choice(['Male', 'Female'], size=n),
        'User_Location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=n),
        'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Gift Card', 'COD'], size=n),
        'Shipping_Method': np.random.choice(['Standard', 'Express', 'Next-Day'], size=n, p=[0.6, 0.25, 0.15]),
    })

    # Ground truth return generation logic
    risk_score = (
        (df['Product_Category'] == 'Clothing').astype(int) * 0.35 +
        (df['Discount_Applied'] >= 0.3).astype(int) * 0.20 +
        (df['Shipping_Method'] == 'Next-Day').astype(int) * 0.10 +
        (df['Product_Price'] > 100).astype(int) * 0.15 +
        np.random.normal(0, 0.2, size=n)
    )
    df['Return_Status'] = (risk_score > 0.45).astype(int)
    df['Days_to_Return'] = np.where(df['Return_Status'] == 1, np.random.randint(1, 30, size=n), 0)
    df['Return_Reason'] = np.where(df['Return_Status'] == 1, 
                                   np.random.choice(['Damaged', 'Wrong Item', 'Changed Mind', 'Fit/Size Issue', 'Late Delivery'], size=n), 
                                   'Not Returned')
    df['Return_Cost'] = np.where(df['Return_Status'] == 1, np.random.uniform(8.0, 28.0, size=n), 0.0)
    df['Profit_Loss'] = np.where(
        df['Return_Status'] == 1,
        -df['Return_Cost'] - (df['Product_Price'] * 0.05),
        df['Product_Price'] * df['Order_Quantity'] * (1 - df['Discount_Applied']) * 0.20
    )

# Normalize target to binary integer
if df['Return_Status'].dtype == object:
    df['Return_Status'] = df['Return_Status'].apply(lambda x: 1 if str(x).strip().lower() in ['returned', '1', 'yes'] else 0)

# -------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA) & STATISTICAL CALCULATIONS
# -------------------------------------------------------------------------
print("=" * 70)
print("SECTION 1: EXPLORATORY DATA ANALYSIS & SUMMARY METRICS")
print("=" * 70)

total_orders = len(df)
total_returns = df['Return_Status'].sum()
overall_return_rate = (total_returns / total_orders) * 100
total_logistics_loss = df['Return_Cost'].sum()
net_profit_loss = df['Profit_Loss'].sum()

print(f"Total Transactions Analyzed    : {total_orders:,}")
print(f"Total Products Returned        : {total_returns:,}")
print(f"Overall Product Return Rate    : {overall_return_rate:.2f}%")
print(f"Cumulative Reverse Logistics Cost: ${total_logistics_loss:,.2f}")
print(f"Net Profit / Loss Impact       : ${net_profit_loss:,.2f}")

# Numerical variables statistical summary
num_summary_cols = ['Product_Price', 'Order_Quantity', 'Discount_Applied', 'User_Age']
print("\n--- Summary Statistics (Numerical Predictors) ---")
print(df[num_summary_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'}).to_string())

# EDA Table 1: Category-wise Breakdown
print("\n--- Return Rate & Financial Impact by Product Category ---")
cat_eda = df.groupby('Product_Category').agg(
    Total_Orders=('Return_Status', 'count'),
    Returned_Orders=('Return_Status', 'sum'),
    Return_Rate_Pct=('Return_Status', lambda x: x.mean() * 100),
    Total_Return_Cost=('Return_Cost', 'sum'),
    Avg_Days_to_Return=('Days_to_Return', lambda x: x[x > 0].mean())
).sort_values(by='Return_Rate_Pct', ascending=False)
print(cat_eda.round(2).to_string())

# EDA Table 2: Return Reasons Breakdown (Returned Subset Only)
returned_df = df[df['Return_Status'] == 1]
print("\n--- Primary Return Reasons Distribution ---")
reason_eda = returned_df['Return_Reason'].value_counts().to_frame(name='Count')
reason_eda['Percentage'] = (reason_eda['Count'] / total_returns) * 100
reason_eda['Cum_Percentage'] = reason_eda['Percentage'].cumsum()
print(reason_eda.round(2).to_string())

# EDA Table 3: Shipping & Payment Method Impact
print("\n--- Return Rate by Shipping Method ---")
shipping_eda = df.groupby('Shipping_Method')['Return_Status'].agg(
    Orders='count',
    Return_Rate=lambda x: f"{x.mean() * 100:.2f}%"
)
print(shipping_eda.to_string())

# Statistical Hypothesis Testing (Chi-Square: Category vs Return Status)
contingency_table = pd.crosstab(df['Product_Category'], df['Return_Status'])
chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test (Category vs Return): Chi2 = {chi2:.2f}, p-value = {p_val:.4e} (Significant: {p_val < 0.05})")

# -------------------------------------------------------------------------
# 3. PREDICTIVE MODELING & PIPELINE SETUP
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 2: MODEL TRAINING & CROSS-VALIDATION")
print("=" * 70)

# Exclude target leakage variables and identifiers
leakage_and_id_cols = [
    'Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date',
    'Return_Reason', 'Days_to_Return', 'Return_Cost', 'Profit_Loss',
    'CO2_Saved', 'Waste_Avoided'
]
feature_cols = [c for c in df.columns if c not in leakage_and_id_cols and c != 'Return_Status']

X = df[feature_cols]
y = df['Return_Status']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
}

# Stratified 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = []
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring_metrics)
    
    cv_results.append({
        'Model': name,
        'CV Accuracy': f"{scores['test_accuracy'].mean():.4f} (±{scores['test_accuracy'].std():.3f})",
        'CV Precision': f"{scores['test_precision'].mean():.4f} (±{scores['test_precision'].std():.3f})",
        'CV Recall': f"{scores['test_recall'].mean():.4f} (±{scores['test_recall'].std():.3f})",
        'CV F1-Score': f"{scores['test_f1'].mean():.4f} (±{scores['test_f1'].std():.3f})",
        'CV ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} (±{scores['test_roc_auc'].std():.3f})"
    })
    
    # Train on full training split
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

print("\n--- 5-Fold Stratified Cross-Validation Results ---")
print(pd.DataFrame(cv_results).to_string(index=False))

# -------------------------------------------------------------------------
# 4. HOLDOUT TEST SET EVALUATION METRICS
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SECTION 3: HOLDOUT TEST SET EVALUATION")
print("=" * 70)

test_eval_records = []

for name, pipe in trained_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_test, y_proba)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    test_eval_records.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sensitivity)': rec,
        'Specificity': specificity,
        'F1-Score': f1,
        'ROC-AUC': auc_val,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    })

eval_df = pd.DataFrame(test_eval_records).sort_values(by='F1-Score', ascending=False)
print("\n--- Model Performance Comparison (Test Split) ---")
print(eval_df[['Model', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity', 'F1-Score', 'ROC-AUC']].round(4).to_string(index=False))

print("\n--- Confusion Matrix Raw Counts (TP, FP, TN, FN) ---")
print(eval_df[['Model', 'TP', 'FP', 'TN', 'FN']].to_string(index=False))

# -------------------------------------------------------------------------
# 5. BEST MODEL DEEP DIVE & FEATURE IMPORTANCE
# -------------------------------------------------------------------------
best_model_name = eval_df.iloc[0]['Model']
best_pipe = trained_pipelines[best_model_name]
y_pred_best = best_pipe.predict(X_test)

print(f"\n--- Detailed Classification Report: {best_model_name} ---")
print(classification_report(y_test, y_pred_best, target_names=['Retained (0)', 'Returned (1)']))

# Feature Importance extraction
cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
all_features = numerical_cols + encoded_cat_cols

classifier_step = best_pipe.named_steps['classifier']
if hasattr(classifier_step, 'feature_importances_'):
    importances = classifier_step.feature_importances_
    feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False)
    print("\n--- Top Predictive Feature Importances ---")
    print(feat_imp.head(8).apply(lambda x: f"{x:.4f}").to_string())
