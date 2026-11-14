import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/online_retail_II.xlsx.csv")

print(f"Combined rows: {len(df):,}")

# Standardize column name ("Customer ID" vs "CustomerID")
if "CustomerID" in df.columns and "Customer ID" not in df.columns:
    df.rename(columns={"CustomerID": "Customer ID"}, inplace=True)

# 2. Basic Statistics
print("\n--- DATASET STATISTICS ---")
print(f"Total Raw Records: {len(df):,}")
print(f"Unique Invoices: {df['Invoice'].nunique():,}")
print(f"Unique Customers: {df['Customer ID'].nunique():,}")
print(f"Unique Products (StockCodes): {df['StockCode'].nunique():,}")
print(f"Cancelled Invoices (Starts with 'C'): {df['Invoice'].astype(str).str.startswith('C').sum():,}")
print(f"Missing Customer IDs: {df['Customer ID'].isnull().sum():,}")

# Clean copy for EDA charts
df_clean = df.dropna(subset=["Customer ID", "Description"]).copy()
df_clean = df_clean[~df_clean["Invoice"].astype(str).str.startswith("C")]
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["Price"] > 0)]
df_clean["TotalSales"] = df_clean["Quantity"] * df_clean["Price"]
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["Description"] = df_clean["Description"].str.strip()

# Plot 1: Top 10 Most Frequently Purchased Products
plt.figure(figsize=(10, 5))
top_products = df_clean["Description"].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis", hue=top_products.index, legend=False)
plt.title("Top 10 Most Frequently Purchased Products")
plt.xlabel("Number of Transactions")
plt.tight_layout()
plt.savefig("plots/top_10_products.png", dpi=300)
plt.close()

# Plot 2: Top 10 Products by Total Revenue
plt.figure(figsize=(10, 5))
top_revenue = df_clean.groupby("Description")["TotalSales"].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_revenue.values, y=top_revenue.index, palette="magma", hue=top_revenue.index, legend=False)
plt.title("Top 10 Products Generating Highest Sales (£)")
plt.xlabel("Total Revenue (£)")
plt.tight_layout()
plt.savefig("plots/top_10_revenue.png", dpi=300)
plt.close()

# Plot 3: Distribution of Transactions Across Top 10 Countries
plt.figure(figsize=(10, 5))
top_countries = df_clean["Country"].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, palette="coolwarm", hue=top_countries.index, legend=False)
plt.title("Top 10 Countries by Transaction Count")
plt.xlabel("Number of Records")
plt.tight_layout()
plt.savefig("plots/top_countries.png", dpi=300)
plt.close()

# Plot 4: Monthly Sales Trend
df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
monthly_sales = df_clean.groupby("YearMonth")["TotalSales"].sum()
plt.figure(figsize=(12, 5))
monthly_sales.plot(marker="o", color="#1f77b4")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plots/monthly_sales_trend.png", dpi=300)
plt.close()

print("\nEDA complete! Charts saved in the 'plots/' folder.")