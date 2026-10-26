import sqlite3 as sql
import pandas as pd

with open("sql/queries.sql", encoding="utf-8") as f:
    queries = [q.strip() for q in f.read().split(";") if q.strip()]

conn = sql.connect("data/instamart.db")
dfs = [pd.read_sql_query(q, conn) for q in queries]
conn.close()

assert len(dfs) == 7, f"expected 7 queries, got {len(dfs)}"

q1, q2, q3, q4, q5, q6, q7 = dfs

print("Q1 rows:", len(q1))                                  
print("Q2 rows:", len(q2), "| total:", round(q2["revenue"].sum(), 2))  
print("Q2 top cell:", q2.iloc[0].tolist())                    
print("Q3 rows:", len(q3))                                    
print("Q4 rows:", len(q4))                                    
print("Q5 rows:", len(q5), "| avg:", q5["avg_customer_spend"].iloc[0])  
print("Q6 top per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]])
print("Q6 rows where rank != row_number:", (q6["rnk"] != q6["row_num"]).sum())   
print("Q7 rows:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1])  
