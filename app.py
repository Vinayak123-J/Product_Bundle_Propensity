import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Product Bundle Propensity Modeler", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/rf_propensity_model.pkl")
    rules = pd.read_pickle("models/association_rules.pkl")
    top_rules = pd.read_pickle("models/top_rules.pkl")
    cust_df = pd.read_pickle("models/customer_features.pkl")
    return model, rules, top_rules, cust_df

st.title("🛒 AI Product Bundle Propensity Modeler")
st.markdown("**FP-Growth Association Rule Mining + Random Forest Purchase Probability Prediction**")

try:
    model, rules, top_rules, cust_df = load_artifacts()
except FileNotFoundError:
    st.error("Models not found! Please run `python train_model.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["🎯 Customer Bundle Recommendations", "📊 Discovered FP-Growth Rules"])

with tab1:
    st.subheader("Predict Bundle Purchase Probability for a Customer")
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_cust = st.selectbox("Select Customer ID", cust_df["Customer ID"].unique())
        cust_row = cust_df[cust_df["Customer ID"] == selected_cust].iloc[0]

        st.metric("Historical Orders (Frequency)", int(cust_row["frequency"]))
        st.metric("Total Spend (Monetary)", f"£{cust_row['monetary']:,.2f}")
        st.metric("Days Since Last Purchase", int(cust_row["recency"]))

    with col2:
        scoring_rows = []
        for _, r in top_rules.iterrows():
            scoring_rows.append({
                "Bundle": f"{r['antecedent_str']}  +  {r['consequent_str']}",
                "Base Product": r["antecedent_str"],
                "Recommended Add-on": r["consequent_str"],
                "recency": cust_row["recency"],
                "frequency": cust_row["frequency"],
                "monetary": cust_row["monetary"],
                "total_items": cust_row["total_items"],
                "unique_products": cust_row["unique_products"],
                "avg_basket_size": cust_row["avg_basket_size"],
                "rule_support": r["support"],
                "rule_confidence": r["confidence"],
                "rule_lift": r["lift"],
                "bought_antecedent": 1,
            })

        score_df = pd.DataFrame(scoring_rows)
        feature_cols = [
            "recency", "frequency", "monetary", "total_items",
            "unique_products", "avg_basket_size", "rule_support",
            "rule_confidence", "rule_lift", "bought_antecedent",
        ]
        probs = model.predict_proba(score_df[feature_cols])[:, 1]
        score_df["Predicted Probability (%)"] = (probs * 100).round(2)

        ranked_bundles = score_df[
            ["Bundle", "Predicted Probability (%)", "rule_lift"]
        ].sort_values(by="Predicted Probability (%)", ascending=False)
        ranked_bundles.rename(columns={"rule_lift": "Association Lift"}, inplace=True)

        st.dataframe(ranked_bundles, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Top Product Associations (FP-Growth)")
    display_rules = rules[
        ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
    ].head(25).copy()
    display_rules.columns = ["If Customer Buys", "Recommend Bundle Item", "Support", "Confidence", "Lift"]
    st.dataframe(display_rules, use_container_width=True, hide_index=True)