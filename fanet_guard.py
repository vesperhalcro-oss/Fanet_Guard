"""
FANET-GUARD : Real-time monitoring of a drone swarm network
-----------------------------------------------------------
Detects NETWORK DROP and SIGNAL JAMMING using a machine learning model,
stores telemetry in PostgreSQL, and falls back to a local CSV backup file
if the database server is down (no flight data is lost).

Usage:
    python fanet_guard.py train                  # train + save ML model
    python fanet_guard.py run                    # run live monitor
    python fanet_guard.py run --simulate-crash   # demo DB crash + recovery
"""

import csv
import os
import random
import sys
import time
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

try:
    import psycopg2
except ImportError:          # demo still works (uses backup file only)
    psycopg2 = None

# ----------------------------- CONFIG ---------------------------------
DB_CONFIG = dict(host="localhost", port=5432, dbname="fanet_guard",
                 user="postgres", password=os.getenv("PGPASSWORD", ""))
NUM_DRONES = 5
TICK_SECONDS = 1.0           # one reading per drone every second
MODEL_FILE = "model.pkl"
BACKUP_FILE = "backup_data.csv"
LABELS = {0: "NORMAL", 1: "NETWORK_DROP", 2: "JAMMING"}

COLUMNS = ["ts", "drone_id", "lat", "lon", "altitude",
           "rssi", "snr", "packet_loss", "latency", "status"]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS telemetry (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMP,
    drone_id    INT,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    altitude    REAL,
    rssi        REAL,        -- signal strength (dBm)
    snr         REAL,        -- signal to noise ratio (dB)
    packet_loss REAL,        -- percent
    latency     REAL,        -- milliseconds
    status      TEXT         -- NORMAL / NETWORK_DROP / JAMMING
);
"""
INSERT = """INSERT INTO telemetry
    (ts, drone_id, lat, lon, altitude, rssi, snr, packet_loss, latency, status)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""


# ------------------------ DATA SIMULATION -----------------------------
def link_sample(label):
    """Return (rssi, snr, packet_loss, latency) for a given link condition."""
    if label == 0:    # healthy link
        rssi, snr = random.gauss(-55, 6), random.gauss(25, 3)
        loss, lat = abs(random.gauss(2, 1.5)), random.gauss(40, 10)
    elif label == 1:  # network drop: weak signal, huge loss/latency
        rssi, snr = random.gauss(-90, 4), random.gauss(8, 3)
        loss, lat = random.uniform(70, 100), random.uniform(500, 1000)
    else:             # jamming: signal looks strong but SNR collapses
        rssi, snr = random.gauss(-52, 8), random.gauss(3, 3)
        loss, lat = random.uniform(40, 90), random.uniform(150, 400)
    return [rssi, snr, min(loss, 100), max(lat, 1)]


class SwarmSimulator:
    """Simulates a drone swarm and randomly injects drops / jamming."""

    def __init__(self, n):
        self.pos = {i: [12.9716 + random.uniform(-.001, .001),
                        77.5946 + random.uniform(-.001, .001),
                        random.uniform(40, 60)] for i in range(1, n + 1)}
        self.events = {}   # drone_id -> [label, ticks_left]

    def step(self):
        if len(self.events) < 2 and random.random() < 0.10:
            d = random.choice(list(self.pos))
            self.events.setdefault(d, [random.choice([1, 2]), random.randint(4, 7)])
        readings = []
        for d, p in self.pos.items():
            p[0] += random.uniform(-2e-5, 2e-5)
            p[1] += random.uniform(-2e-5, 2e-5)
            p[2] += random.uniform(-0.5, 0.5)
            label = self.events[d][0] if d in self.events else 0
            if d in self.events:
                self.events[d][1] -= 1
                if self.events[d][1] <= 0:
                    del self.events[d]
            readings.append((d, p[0], p[1], p[2], link_sample(label), label))
        return readings


# --------------------------- ML MODEL ---------------------------------
def train():
    X, y = [], []
    for label in LABELS:
        for _ in range(3000):
            X.append(link_sample(label))
            y.append(label)
    X_tr, X_te, y_tr, y_te = train_test_split(np.array(X), np.array(y),
                                              test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te),
                                target_names=list(LABELS.values())))
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")


# ----------------------- STORAGE (DB + BACKUP) ------------------------
def db_insert(rows, force_down=False):
    """Insert rows into PostgreSQL. Raises an error if DB is unreachable."""
    if force_down or psycopg2 is None:
        raise ConnectionError("database unavailable")
    conn = psycopg2.connect(connect_timeout=2, **DB_CONFIG)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE)
            cur.executemany(INSERT, rows)
    finally:
        conn.close()


def backup_write(rows):
    new_file = not os.path.exists(BACKUP_FILE)
    with open(BACKUP_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLUMNS)
        w.writerows(rows)


def backup_restore():
    """Push data saved in the backup file into PostgreSQL, then delete it."""
    if not os.path.exists(BACKUP_FILE):
        return 0
    with open(BACKUP_FILE, newline="") as f:
        rows = list(csv.reader(f))[1:]
    if rows:
        db_insert(rows)
        os.remove(BACKUP_FILE)
    return len(rows)


# --------------------------- LIVE MONITOR -----------------------------
def run(simulate_crash=False):
    if not os.path.exists(MODEL_FILE):
        sys.exit("Model not found. Run:  python fanet_guard.py train")
    model = joblib.load(MODEL_FILE)
    swarm = SwarmSimulator(NUM_DRONES)
    tick = 0
    print("FANET-GUARD started. Press Ctrl+C to stop.\n")

    try:
        while True:
            tick += 1
            rows, alerts = [], []
            for d, lat, lon, alt, feats, _truth in swarm.step():
                pred = int(model.predict([feats])[0])
                status = LABELS[pred]
                rows.append((datetime.now().isoformat(sep=" ", timespec="seconds"),
                             d, round(lat, 6), round(lon, 6), round(alt, 1),
                             round(feats[0], 1), round(feats[1], 1),
                             round(feats[2], 1), round(feats[3], 1), status))
                if pred != 0:
                    alerts.append(f"   !! ALERT  Drone {d}: {status}  "
                                  f"(RSSI {feats[0]:.0f} dBm, SNR {feats[1]:.0f} dB, "
                                  f"loss {feats[2]:.0f}%)")

            # demo: fake a server crash between ticks 8 and 17
            crashed = simulate_crash and 8 <= tick <= 17
            try:
                db_insert(rows, force_down=crashed)
                restored = backup_restore()
                msg = f"[DB OK]     saved {len(rows)} rows"
                if restored:
                    msg += f"  | recovered {restored} rows from backup file"
            except Exception:
                backup_write(rows)
                msg = f"[DB DOWN]   saved {len(rows)} rows to {BACKUP_FILE}"

            print(f"Tick {tick:03d} {msg}")
            for a in alerts:
                print(a)
            time.sleep(TICK_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train()
    elif cmd == "run":
        run(simulate_crash="--simulate-crash" in sys.argv)
    else:
        print(__doc__)
