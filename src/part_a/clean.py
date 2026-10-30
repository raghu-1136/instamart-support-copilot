import pandas as pd
import numpy as np
import os
import sqlite3 as sql

customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

orders = orders.merge(products[["product_id", "category","price_inr"]], on="product_id", how="left")

# Standardize city to Title Case after trimming whitespace to collapse 24 raw variations to 6 canonical metro cities
customers["city"] = customers["city"].str.strip().str.title()


clean_gender = {"female":"female","FEMALE":"female","F":"female","Female":"female","f":"female",
                "male":"male","MALE":"male","M":"male","Male":"male","m":"male"}
customers["gender"] = customers["gender"].map(clean_gender)

customers["gender"] = customers["gender"].fillna("unknown")

out_of_range = (customers["age"] < 10) | (customers["age"] > 100)
n_out = out_of_range.sum()
customers.loc[out_of_range, "age"] = np.nan

# Impute missing/invalid ages using the customer's city median; preserves regional demographic distributions
n_to_fill = customers["age"].isna().sum()
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
print("Out-of-range ages set to missing:", n_out)
print("Ages filled with city median:", n_to_fill)


customers["signup_date"] = pd.to_datetime(customers["signup_date"])

orders["order_ts"] = pd.to_datetime(orders["order_ts"])


n_before = len(orders)
orders = orders.drop_duplicates()
n_exact = n_before - len(orders)
print("Exact duplicates removed:", n_exact)

# Retain only the latest record per order_id by order_ts to discard preliminary 'pending' duplicates
n_before = len(orders)
orders = (orders.sort_values("order_ts")
                .drop_duplicates(subset="order_id", keep="last")
                .reset_index(drop=True))
n_older = n_before - len(orders)
print("Older records removed (keep latest per order_id):", n_older)

orders["order_amount"] = pd.to_numeric(orders["order_amount"],errors = "coerce")

# Recompute missing/corrupt order amounts using catalog price: round(price_inr * qty * (1 - discount/100), 2)
n_amt = orders["order_amount"].isna().sum()

orders["order_amount"] = orders["order_amount"].fillna(
    (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
)

print("Amounts filled from price:", n_amt)

n_neg = (orders["delivery_minutes"] < 0).sum()

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
print("Negative delivery times set to missing:", n_neg)

# Category-median imputation for completed orders; cancelled orders are intentionally kept NaN as they were never dispatched
completed = orders["status"] == "completed"

dm_completed = orders["delivery_minutes"].where(completed)
cat_median_per_row = dm_completed.groupby(orders["category"]).transform("median")

fill_mask = completed & orders["delivery_minutes"].isna()
print("Delivery minutes filled:", fill_mask.sum())

orders.loc[fill_mask, "delivery_minutes"] = cat_median_per_row[fill_mask]



orders = orders.sort_values("order_ts").reset_index(drop=True)
orders = orders.drop(columns=["category", "price_inr"])
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")


customers_db = customers.assign(signup_date=customers["signup_date"].dt.strftime("%Y-%m-%d"))


os.makedirs("data/clean", exist_ok=True)

def store(df, name, conn):
    df.to_csv(f"data/clean/{name}.csv", index=False)
    df.to_sql(name, conn, if_exists="replace", index=False)

conn = sql.connect("data/instamart.db")
store(products, "products", conn)
store(orders, "orders", conn)
store(customers_db, "customers", conn)

conn.close()
