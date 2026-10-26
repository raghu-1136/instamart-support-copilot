import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
#----------audit tables-----------
def audit_table(name,df):
    print(f"\n==={name}===")
    print("Shape",df.shape)

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean()*100).round(2),
        })
    print(summary)

audit_table("orders", orders)
audit_table("customers",customers)
audit_table("products", products)

print("\nRaw orders rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print("Pending rows:", (orders["status"].str.strip().str.lower() == "pending").sum())
print("Distinct order_id:", orders["order_id"].nunique())

# --- orders ---
amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
print("\nNon-numeric order_amount:", amount_numeric.isna().sum())
print("Negative delivery_minutes:", (orders["delivery_minutes"] < 0).sum())
print("Blank delivery_minutes:", orders["delivery_minutes"].isna().sum())

# --- customers ---
print("\nBlank age:", customers["age"].isna().sum())
print("Age outside 10-100:", ((customers["age"] < 10) | (customers["age"] > 100)).sum())
print("Distinct raw city spellings:", customers["city"].nunique())
print("Distinct non-blank gender spellings:", customers["gender"].nunique())
print("Blank gender:", customers["gender"].isna().sum())
