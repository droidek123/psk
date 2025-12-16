import os
import glob
import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# -------------------------------------------------------------
# GRAF – WCZYTANIE TOPOLOGII
# -------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    """
    Oczekiwany format:
    line 0: liczba węzłów
    line 1: (ignorowane)
    dalej: macierz sąsiedztwa
    """
    with open(path) as f:
        lines = f.read().strip().splitlines()

    n = int(lines[0])
    mat = np.loadtxt(lines[2:], dtype=float)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            w = mat[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)

    return G


def precompute_graph_stats(G: nx.Graph) -> dict:
    return {
        "degree": dict(G.degree()),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness": nx.closeness_centrality(G, distance="weight"),
    }


def extract_graph_request_features(
    df: pd.DataFrame,
    G: nx.Graph,
    gstats: dict,
) -> dict:

    path_lengths = []
    hop_counts = []
    src_deg = []
    dst_deg = []
    src_bet = []
    dst_bet = []
    max_edge_weights = []   # NEW

    for _, r in df.iterrows():
        s, d = int(r.source), int(r.destination)

        try:
            path = nx.shortest_path(G, s, d, weight="weight")
            length = nx.shortest_path_length(G, s, d, weight="weight")

            path_lengths.append(length)
            hop_counts.append(len(path) - 1)

            src_deg.append(gstats["degree"][s])
            dst_deg.append(gstats["degree"][d])
            src_bet.append(gstats["betweenness"][s])
            dst_bet.append(gstats["betweenness"][d])

            # -------- NEW FEATURE --------
            edge_weights = [
                G[path[i]][path[i + 1]]["weight"]
                for i in range(len(path) - 1)
            ]
            max_edge_weights.append(max(edge_weights))
            # --------------------------------

        except nx.NetworkXNoPath:
            continue

    if not path_lengths:
        return {}

    return {
        "spath_len_mean": np.mean(path_lengths),
        "spath_len_std": np.std(path_lengths),
        "hop_count_mean": np.mean(hop_counts),
        "src_degree_mean": np.mean(src_deg),
        "dst_degree_mean": np.mean(dst_deg),
        "src_betweenness_mean": np.mean(src_bet),
        "dst_betweenness_mean": np.mean(dst_bet),
        "max_edge_weight_on_path_mean": np.mean(max_edge_weights),  # NEW
    }

# -------------------------------------------------------------
# 1. CECHY Z requests.csv
# -------------------------------------------------------------
def extract_features(df: pd.DataFrame) -> pd.Series:
    feats = {}

    feats["n_requests"] = len(df)
    feats["bitrate_sum"] = df["bitrate"].sum()
    feats["bitrate_mean"] = df["bitrate"].mean()
    feats["bitrate_std"] = df["bitrate"].std()
    feats["bitrate_min"] = df["bitrate"].min()
    feats["bitrate_max"] = df["bitrate"].max()

    feats["n_unique_sources"] = df["source"].nunique()
    feats["n_unique_destinations"] = df["destination"].nunique()
    feats["n_unique_pairs"] = df[["source", "destination"]].drop_duplicates().shape[0]

    feats["mean_requests_per_source"] = df["source"].value_counts().mean()
    feats["mean_requests_per_destination"] = df["destination"].value_counts().mean()

    return pd.Series(feats)


# -------------------------------------------------------------
# 2. WCZYTANIE DANYCH JEDNEJ TOPOLOGII
# -------------------------------------------------------------
def load_topology(
    path: str,
    topology_name: str,
    graph_file: str,
) -> pd.DataFrame:
    
    G = load_graph(graph_file)
    gstats = precompute_graph_stats(G)

    rows = []

    # np. RSA_estimation/Euro28/request-set_0
    request_sets = sorted(glob.glob(os.path.join(path, "request-set_*")))

    for rs in request_sets:
        req_path = os.path.join(rs, "requests.csv")
        res_path = os.path.join(rs, "results.txt")

        if not (os.path.exists(req_path) and os.path.exists(res_path)):
            continue

        # --- requests.csv ---
        df_req = pd.read_csv(req_path)

        # --- results.txt ---
        metrics = {}
        with open(res_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # format: nazwa SPACJA wartość (np. "highestSlot 99.0")
                parts = line.split()
                if len(parts) < 2:
                    continue

                key = parts[0]
                val_str = parts[-1].replace(",", ".")
                try:
                    value = float(val_str)
                except ValueError:
                    continue

                metrics[key] = value

        feat = extract_features(df_req)

        graph_feats = extract_graph_request_features(df_req, G, gstats)
        for k, v in graph_feats.items():
            feat[k] = v

        print (feat)


        target_keys = [
            "highestSlot",
            "avgHighestSlot",
            "sumOfSlots",
            "avgActiveTransceivers",
        ]
        for key in target_keys:
            feat[key] = metrics.get(key, None)

        feat["topology"] = topology_name
        feat["request_set"] = os.path.basename(rs)

        rows.append(feat)

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# 3. WCZYTANIE EURO28 + US26
# -------------------------------------------------------------
base = "RSA_estimation"

df_euro = load_topology(os.path.join(base, "Euro28"), "Euro28", os.path.join(base, "Euro28", "euro28.net"))
df_us = load_topology(os.path.join(base, "US26"), "US26", os.path.join(base, "US26", "us26.net"))

data = pd.concat([df_euro, df_us], ignore_index=True)

target_cols = [
    "highestSlot",
    "avgHighestSlot",
    "sumOfSlots",
    "avgActiveTransceivers",
]

# wyrzucamy wiersze bez metryk
data = data.dropna(subset=target_cols)

print("Wczytane dane (pierwsze wiersze):")
print(data.head())
print("Liczba przykładów:", len(data))


# -------------------------------------------------------------
# 4. PODZIAŁ NA X / y ORAZ TRAIN / TEST
# -------------------------------------------------------------
feature_cols = [
    c for c in data.columns
    if c not in target_cols + ["topology", "request_set"]
]

X = data[feature_cols]
y = data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------------------------------------------
# 5. MODELE XGBoost I LightGBM (multi-output przez MultiOutputRegressor)
# -------------------------------------------------------------
models = {
    "XGBoost": MultiOutputRegressor(
        XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    ),
    "LightGBM": MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
    ),
}


def evaluate(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds, multioutput="raw_values")
    r2 = r2_score(y_test, preds, multioutput="raw_values")

    print(f"\n===== {name} =====")
    for i, col in enumerate(target_cols):
        print(f"{col:22s} | MAE={mae[i]:8.3f}  | R2={r2[i]:6.3f}")


# trenowanie XGBoost i LightGBM
for name, model in models.items():
    evaluate(name, model)


# -------------------------------------------------------------
# 6. CATBOOST – multioutput bez MultiOutputRegressor
# -------------------------------------------------------------
print("\n===== CatBoost (MultiRMSE) =====")

cat_model = CatBoostRegressor(
    depth=6,
    learning_rate=0.05,
    n_estimators=400,
    loss_function="MultiRMSE",  # obsługa wielu wyjść
    verbose=False,
    random_seed=42,
)

cat_model.fit(X_train, y_train.values)

y_pred_cb = cat_model.predict(X_test)

mae_cb = mean_absolute_error(y_test, y_pred_cb, multioutput="raw_values")
r2_cb = r2_score(y_test, y_pred_cb, multioutput="raw_values")

for i, col in enumerate(target_cols):
    print(f"{col:22s} | MAE={mae_cb[i]:8.3f}  | R2={r2_cb[i]:6.3f}")


print("\nGotowe.")