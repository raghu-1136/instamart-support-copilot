import pandas as pd
import sqlite3 as sql

conn = sql.connect("data/instamart.db")
assert pd.read_sql_query("SELECT COUNT(*) AS n FROM orders", conn)["n"][0] == 12309
conn.close()

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

expected_cities = {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                   "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
assert customers["city"].value_counts().to_dict() == expected_cities

expected_genders = {"female":610,"male":560,"unknown":30}
assert customers["gender"].value_counts().to_dict() ==expected_genders

assert customers["age"].isna().sum() == 0
assert customers["age"].between(18, 60).all()

assert len(orders) == 12309, f"expected 12309 orders, got {len(orders)}"

completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

assert completed.sum() == 11110
assert cancelled.sum() == 1199

assert orders["order_id"].is_unique
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].notna().all()

assert orders.loc[completed, "delivery_minutes"].notna().all()
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199

total = orders.loc[completed, "order_amount"].sum()
assert abs(total - 2998668.80) < 0.01, f"revenue was {total}"

print("ALL CHECKPOINTS PASSED")
