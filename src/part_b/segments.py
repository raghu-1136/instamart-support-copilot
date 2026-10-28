import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

full_orders = pd.read_csv("data/clean/orders.csv")

completed = full_orders[full_orders["status"] == "completed"].copy()

completed["order_ts"] = pd.to_datetime(completed["order_ts"])

rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
)

snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

rfm = rfm.drop(columns="last_order_ts").reset_index()

rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(len(rfm))                                   # 1175
print(round(rfm["recency"].median(),2),round(rfm["recency"].min(),2),round(rfm["recency"].max(),2) )                # median 19, min 0, max 180
print(round(rfm["frequency"].mean(),2), rfm["frequency"].max())   # 9.46, 33
print(round(rfm["monetary"].mean(),2), round(rfm["monetary"].max(),2)) 

features = ["recency",
            "frequency",
            "monetary"]
X = rfm[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rows = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,   # Number of clusters
        random_state=42, # Reproducible initialization
        n_init=10         # Run multiple initializations and keep the best result
    )
    labels = kmeans.fit_predict(X_scaled)
    rows.append({
        "K": k,
        "inertia": kmeans.inertia_,
        "silhouette": silhouette_score(X_scaled, labels),
    })

k_table = pd.DataFrame(rows).round(3)
print(k_table)


kmeans = KMeans(
    n_clusters=4,   # We decided to use 4 customer segments
    random_state=42, # Reproducibility
    n_init=10        # Multiple initializations
)

rfm["segment_id"] = kmeans.fit_predict(X_scaled)

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    recency=("recency", "mean"),
    frequency=("frequency", "mean"),
    monetary=("monetary", "mean"),
).round(2)
print(profile)

pca = PCA(n_components=2)
pca.fit(X_scaled)
print(pca.explained_variance_ratio_, pca.explained_variance_ratio_.sum())

premium_id = profile["monetary"].idxmax()
loyal_id = profile["frequency"].idxmax()
dormant_id = profile["recency"].idxmax()
assert len({premium_id, loyal_id, dormant_id}) == 3

names = {sid: "Occasional Shoppers" for sid in profile.index}
names[premium_id] = "Premium Big Spenders"
names[loyal_id] = "Loyal Regulars"
names[dormant_id] = "Lapsing / Dormant"
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id

print("Premium segment id:", premium_id)
print("PCA variance explained:", pca.explained_variance_ratio_.round(3), round(pca.explained_variance_ratio_.sum(), 3))

assert len(rfm) == 1175
assert sorted(profile["size"].tolist()) == [138, 182, 419, 436]
assert rfm["is_premium"].sum() == 138

os.makedirs("outputs", exist_ok=True)
out = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]]
out.to_csv("outputs/customer_segments.csv", index=False)
