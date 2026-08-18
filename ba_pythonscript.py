import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# -------------------------------------------------------------------------
# Set Theme & Output Settings
# -------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# -------------------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------------------
# Place your downloaded CSV in the same folder or update the path below:
dataset_path = "returns_management.csv"

if os.path.exists(dataset_path):
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
else:
    print(f"File '{dataset_path}' not found locally. Simulating Kaggle schema structure...")
    np.random.seed(42)
    n = 5000
    df = pd.DataFrame({
        'Order_ID': [f'ORD_{i:05d}' for i in range(1, n + 1)],
        'Product_ID': [f'PROD_{np.random.randint(100, 500)}' for _ in range(n)],
        'User_ID': [f'USR_{np.random.randint(1000, 3000)}' for _ in range(n)],
        'Order_Date': pd.date_range('2025-01-01', periods=n, freq='h'),
        'Product_Category': np.random.choice(['Clothing', 'Electronics', 'Books', 'Toys', 'Home'], size=n, p=[0.35, 0.25, 0.15, 0.1, 0.15]),
        'Product_Price': np.random.exponential(scale=60, size=n) + 15,
        'Order_Quantity': np.random.choice([1, 2, 3, 4], size=n, p=[0.65, 0.2, 0.1, 0.05]),
        'Discount_Applied': np.random.choice([0.0, 0.1, 0.2, 0.3, 0.5], size=n),
        'Return_Status': np.random.choice([0, 1], size=n, p=[0.72, 0.28]),
        'Return_Reason': np.random.choice(['Damaged', 'Wrong Item', 'Changed Mind', 'Fit/Size Issue', 'Late Delivery'], size=n),
        'Days_to_Return': np.random.randint(1, 30, size=n),
        'User_Age': np.random.randint(18, 68, size=n),
        'User_Gender': np.random.choice(['Male', 'Female'], size=n),
        'User_Location': np.random.choice(['Urban', 'Suburban', 'Rural'], size=n),
        'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Gift Card'], size=n),
        'Shipping_Method': np.random.choice(['Standard', 'Express', 'Next-Day'], size=n, p=[0.6, 0.25, 0.15]),
        'Return_Cost': np.random.uniform(5.0, 30.0, size=n),
        'Profit_Loss': np.random.uniform(-50.0, 150.0, size=n)
    })
    df.loc[df['Return_Status'] == 0, 'Days_to_Return'] = 0
    df.loc[df['Return_Status'] == 0, 'Return_Reason'] = 'Not Returned'
    df.loc[df['Return_Status'] == 0, 'Return_Cost'] = 0.0

# Normalize target variable if formatted as text
if df['Return_Status'].dtype == object:
    df['Return_Status'] = df['Return_Status'].apply(lambda x: 1 if str(x).strip().lower() in ['returned', '1', 'yes'] else 0)

# -------------------------------------------------------------------------
# 2. FEATURE ENGINEERING & PREPROCESSING
# -------------------------------------------------------------------------
# Exclude post-return leakage fields and high-cardinality IDs
leakage_cols = [
    'Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date',
    'Return_Reason', 'Days_to_Return', 'Return_Cost', 'Profit_Loss',
    'CO2_Saved', 'Waste_Avoided'
]
feature_cols = [c for c in df.columns if c not in leakage_cols and c != 'Return_Status']

X = df[feature_cols]
y = df['Return_Status']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Predictor features ({len(feature_cols)}): {feature_cols}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------------------
# 3. MODEL TRAINING & PIPELINES
# -------------------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
}

trained_pipes = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    trained_pipes[name] = pipe

best_pipe = trained_pipes['Random Forest']
y_pred = best_pipe.predict(X_test)
returned_subset = df[df['Return_Status'] == 1]

# -------------------------------------------------------------------------
# 4. GENERATE 6 SEPARATE PROJECT VISUALIZATIONS
# -------------------------------------------------------------------------

# Graph 1: Category Return Rates
plt.figure(figsize=(7.5, 4.5))
cat_order = df.groupby('Product_Category')['Return_Status'].mean().sort_values(ascending=False).index
ax = sns.barplot(data=df, x='Product_Category', y='Return_Status', order=cat_order, errorbar=None, palette='Blues_r')
plt.title('Return Rate by Product Category (%)', fontweight='bold', pad=12)
plt.ylabel('Return Rate')
plt.xlabel('Category')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y_val, _: f'{y_val*100:.0f}%'))
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='semibold')
plt.tight_layout()
plt.savefig('graph1_category_return_rate.png', dpi=300)
plt.show()

# Graph 2: Return Reason Breakdown
plt.figure(figsize=(8, 4.5))
valid_reasons = returned_subset[returned_subset['Return_Reason'] != 'Not Returned']['Return_Reason'].value_counts()
sns.barplot(x=valid_reasons.values, y=valid_reasons.index, palette='crest')
plt.title('Primary Reasons for Product Returns', fontweight='bold', pad=12)
plt.xlabel('Number of Orders Returned')
plt.tight_layout()
plt.savefig('graph2_return_reasons.png', dpi=300)
plt.show()

# Graph 3: Days to Return Distribution
plt.figure(figsize=(7.5, 4))
sns.histplot(returned_subset[returned_subset['Days_to_Return'] > 0]['Days_to_Return'], bins=15, kde=True, color='#2c5282')
plt.title('Distribution of Return Turnaround (Days to Return)', fontweight='bold', pad=12)
plt.xlabel('Days Elapsed Since Purchase')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('graph3_days_to_return.png', dpi=300)
plt.show()

# Graph 4: Confusion Matrix
plt.figure(figsize=(5.5, 4.5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Retained (0)', 'Returned (1)'],
            yticklabels=['Retained (0)', 'Returned (1)'])
plt.title('Confusion Matrix (Random Forest Classifier)', fontweight='bold', pad=12)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('graph4_confusion_matrix.png', dpi=300)
plt.show()

# Graph 5: Multi-Model ROC Curve
plt.figure(figsize=(7, 5))
for name, pipe in trained_pipes.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curves Across Predictive Models', fontweight='bold', pad=12)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('graph5_roc_curves.png', dpi=300)
plt.show()

# Graph 6: Top Feature Importance
cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
all_features = numerical_cols + encoded_cat_cols
importances = best_pipe.named_steps['classifier'].feature_importances_
feat_series = pd.Series(importances, index=all_features).sort_values(ascending=True)

plt.figure(figsize=(8, 4.5))
feat_series.tail(8).plot(kind='barh', color='#3182bd')
plt.title('Top 8 Predictors of Return Risk', fontweight='bold', pad=12)
plt.xlabel('Relative Feature Importance Score')
plt.tight_layout()
plt.savefig('graph6_feature_importance.png', dpi=300)
plt.show()
