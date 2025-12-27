import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

CONFIG = {
    "days": 365,
    "junctions": 6,
    "seq_len": 24,
    "test_size": 0.2,
    "epochs": 25,
    "batch": 64
}

def banner(t):
    print("\n" + "=" * 80)
    print(t.center(80))
    print("=" * 80)

def metrics(n, y, p):
    print(n)
    print("MAE :", round(mean_absolute_error(y, p), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y, p)), 2))
    print("R2  :", round(r2_score(y, p), 3))
    print("-" * 60)

class Generator:
    def create(self):
        rows = CONFIG["days"] * 24 * CONFIG["junctions"]
        idx = np.arange(rows)

        hour = idx % 24
        day = (idx // 24) % 30 + 1
        month = (idx // (24 * 30)) % 12 + 1
        weekday = idx % 7
        weekend = (weekday >= 5).astype(int)

        junction = np.tile(
            np.arange(1, CONFIG["junctions"] + 1),
            rows // CONFIG["junctions"]
        )

        weather = np.random.choice(
            ["sunny", "rainy", "cloudy", "foggy"],
            rows,
            p=[0.45, 0.25, 0.2, 0.1]
        )

        accident = np.random.choice([0, 1], rows, p=[0.96, 0.04])

        rush = (np.sin(hour * np.pi / 12) + 1) * 100
        seasonal = (np.sin(month * np.pi / 6) + 1) * 60
        noise = np.random.randint(0, 80, rows)

        traffic = np.abs(rush + seasonal + noise).astype(int)

        return pd.DataFrame({
            "junction_id": junction,
            "hour": hour,
            "day": day,
            "month": month,
            "weekday": weekday,
            "is_weekend": weekend,
            "weather": weather,
            "accident": accident,
            "traffic_volume": traffic
        })

class Preprocess:
    def __init__(self):
        self.enc = LabelEncoder()
        self.scaler = MinMaxScaler()

    def run(self, df):
        df["weather"] = self.enc.fit_transform(df["weather"])
        X = df.drop("traffic_volume", axis=1)
        y = df["traffic_volume"]
        X = self.scaler.fit_transform(X)
        return train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)

class ML:
    def lr(self): return LinearRegression()
    def dt(self): return DecisionTreeRegressor(max_depth=10)
    def rf(self): return RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1)
    def xgb(self): return XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.07)

def make_seq(X, y, l):
    xs, ys = [], []
    for i in range(len(X) - l):
        xs.append(X[i:i+l])
        ys.append(y.iloc[i+l])
    return np.array(xs), np.array(ys)

def plot_result(y, p):
    plt.figure(figsize=(12,5))
    plt.plot(y[:200], label="Actual")
    plt.plot(p[:200], label="Predicted")
    plt.legend()
    plt.xlabel("Time Index")
    plt.ylabel("Traffic Volume")
    plt.title("Traffic Pattern Prediction")
    plt.show()

def traffic_map(df):
    coords = {
        1:(28.6139,77.2090),
        2:(28.5355,77.3910),
        3:(28.4595,77.0266),
        4:(28.7041,77.1025),
        5:(28.4089,77.3178),
        6:(28.6692,77.4538)
    }

    m = folium.Map(location=[28.61,77.20], zoom_start=10)
    avg = df.groupby("junction_id")["traffic_volume"].mean()

    for j,v in avg.items():
        folium.CircleMarker(
            location=coords[j],
            radius=min(20, v/25),
            popup=f"Junction {j}<br>Avg Traffic {int(v)}",
            color="red",
            fill=True
        ).add_to(m)

    m.save("traffic_map.html")

def main():
    banner("TRAFFIC PATTERN PREDICTION FOR SMART CITIES")

    df = Generator().create()
    prep = Preprocess()
    Xtr, Xte, ytr, yte = prep.run(df)

    ml = ML()
    models = {
        "Linear Regression": ml.lr(),
        "Decision Tree": ml.dt(),
        "Random Forest": ml.rf(),
        "XGBoost": ml.xgb()
    }

    for n,m in models.items():
        banner(f"Training {n}")
        m.fit(Xtr, ytr)
        metrics(n, yte, m.predict(Xte))

    Xall = prep.scaler.transform(df.drop("traffic_volume", axis=1))
    yall = df["traffic_volume"]

    Xs, ys = make_seq(Xall, yall, CONFIG["seq_len"])
    s = int(0.8 * len(Xs))

    Xtr_s, Xte_s = Xs[:s], Xs[s:]
    ytr_s, yte_s = ys[:s], ys[s:]

    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(Xtr_s.shape[1], Xtr_s.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    lstm.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=5, restore_best_weights=True)

    lstm.fit(
        Xtr_s,
        ytr_s,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch"],
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    preds = lstm.predict(Xte_s).flatten()
    metrics("LSTM", yte_s, preds)

    plot_result(yte_s, preds)
    traffic_map(df)

    sample = np.array([[3,18,15,7,2,0,1,0]])
    sample = prep.scaler.transform(sample)
    print("Predicted Traffic Volume:", int(models["XGBoost"].predict(sample)[0]))

if __name__ == "__main__":
    main()
