import pandas as pd
import numpy as np

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")

completed = orders[orders["status"] == "completed"]

merged = (completed[["customer_id", "product_id", "order_amount"]]
          .merge(customers[["customer_id", "city"]], on="customer_id", how="left")
          .merge(products[["product_id", "category"]], on="product_id", how="left"))

pvt_tbl = pd.pivot_table(
    merged,
    index = "city",
    columns = "category",
    values = "order_amount",
    aggfunc ="sum",
    margins = True)

assert abs(pvt_tbl.loc["All", "All"] - 2998668.80) < 0.01

print("Pivot grand total:", round(pvt_tbl.loc["All", "All"], 2))

city_stats = merged.groupby("city").agg(
    total=("order_amount", "sum"),
    average=("order_amount", "mean"),
    orders=("order_amount", "count"),
).round(2)

print(city_stats)

amt = merged["order_amount"]

mean, median = amt.mean(), amt.median()
skew = amt.skew()
q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

print(f"mean {mean:.2f}, median {median:.2f}, skewness {skew:.2f}")
print(f"Q1 {q1:.2f}, Q3 {q3:.2f}, IQR {iqr:.2f}")
print(f"fences: lower {lower:.2f}, upper {upper:.2f}")
print("outliers above:", (amt > upper).sum())
print("outliers below:", (amt < lower).sum())

cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = orders.loc[orders["status"] == "completed", cols].corr()
print(corr.round(3))

pairs = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
print(pairs.sort_values(ascending=False).head(2))

upper_outliers = merged[merged["order_amount"] > upper]
print(upper_outliers["category"].value_counts())

print(upper_outliers[upper_outliers["category"] == "Dairy"]["product_id"].value_counts())
