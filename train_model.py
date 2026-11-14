import os
import joblib
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)

os.makedirs("models", exist_ok=True)

# ==========================================
# STEP 1: DATA LOADING & CLEANING
# ==========================================
print("1. Loading and cleaning dataset...")
df = pd.read_csv("data/online_retail_II.xlsx.csv")

if "CustomerID" in df.columns and "Customer ID" not in df.columns:
    df.rename(columns={"CustomerID": "Customer ID"}, inplace=True)

df.drop_duplicates(inplace=True)
df.dropna(subset=["Customer ID", "Description"], inplace=True)
df = df[~df["Invoice"].astype(str).str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

df["Description"] = df["Description"].str.strip()
df["Customer ID"] = df["Customer ID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalAmount"] = df["Quantity"] * df["Price"]

# Focus on UK + top 300 products
df_uk = df[df["Country"] == "United Kingdom"].copy()
top_items = df_uk["Description"].value_counts().head(300).index
df_filtered = df_uk[df_uk["Description"].isin(top_items)].copy()

# Temporal split
split_date = "2011-09-01"
train_df = df_filtered[df_filtered["InvoiceDate"] < split_date].copy()
test_df = df_filtered[df_filtered["InvoiceDate"] >= split_date].copy()
print(f"Train records: {len(train_df):,} | Test records: {len(test_df):,}")

# ==========================================
# STEP 2: FP-GROWTH ASSOCIATION RULES
# ==========================================
print("\n2. Running FP-Growth...")
baskets = train_df.groupby("Invoice")["Description"].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(baskets).transform(baskets)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(basket_df, min_support=0.015, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

rules["antecedent_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequent_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
# Keep only 1-to-1 rules so bundles are clean
rules = rules[
    (rules["antecedents"].apply(len) == 1) &
    (rules["consequents"].apply(len) == 1)
]
rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)
print(f"Discovered {len(rules)} high-lift rules")
print(rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].head(5))

rules.to_pickle("models/association_rules.pkl")

# ==========================================
# STEP 3: RANDOM FOREST PROPENSITY MODEL
# ==========================================
print("\n3. Building customer features & training RF...")

def build_customer_features(data):
    ref_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    return data.groupby("Customer ID").agg(
        recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        frequency=("Invoice", "nunique"),
        monetary=("TotalAmount", "sum"),
        total_items=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
        avg_basket_size=("Quantity", "mean"),
    ).reset_index()

train_cust = build_customer_features(train_df)
top_rules = rules.head(15).copy()

def create_propensity_dataset(source_df, cust_features, rule_subset):
    cust_items = source_df.groupby("Customer ID")["Description"].apply(set).to_dict()
    rows = []
    for cust_id, purchased in cust_items.items():
        for idx, rule in rule_subset.iterrows():
            ant = set(rule["antecedents"])
            cons = set(rule["consequents"])
            bought_antecedent = int(ant.issubset(purchased))
            target_bundle = int(ant.union(cons).issubset(purchased))
            rows.append({
                "Customer ID": cust_id,
                "rule_id": idx,
                "rule_support": rule["support"],
                "rule_confidence": rule["confidence"],
                "rule_lift": rule["lift"],
                "bought_antecedent": bought_antecedent,
                "target": target_bundle,
            })
    pair_df = pd.DataFrame(rows)
    return pair_df.merge(cust_features, on="Customer ID", how="inner")

train_data = create_propensity_dataset(train_df, train_cust, top_rules)
test_data = create_propensity_dataset(test_df, train_cust, top_rules)

feature_cols = [
    "recency", "frequency", "monetary", "total_items",
    "unique_products", "avg_basket_size", "rule_support",
    "rule_confidence", "rule_lift", "bought_antecedent",
]

X_train, y_train = train_data[feature_cols], train_data["target"]
X_test, y_test = test_data[feature_cols], test_data["target"]

rf_model = RandomForestClassifier(
    n_estimators=150, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf_model.fit(X_train, y_train)

# ==========================================
# STEP 4: EVALUATION
# ==========================================
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n--- RANDOM FOREST METRICS ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(rf_model, "models/rf_propensity_model.pkl")
train_cust.to_pickle("models/customer_features.pkl")
top_rules.to_pickle("models/top_rules.pkl")
print("\nSaved model and artifacts to 'models/' successfully!")