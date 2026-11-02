# FANET-GUARD
Real-time drone swarm network monitor: detects **network drops** and **signal jamming**
with a Python ML model, stores data in **PostgreSQL**, and auto-saves to a **local CSV backup**
if the database server crashes.

## Architecture
Drone Swarm Simulator -> ML Model (Random Forest) -> PostgreSQL
                                                  \-> backup_data.csv (if DB is down)

## Setup (one time)
1. Install Python 3.9+ and PostgreSQL.
2. Create the database: `createdb -U postgres fanet_guard`
   (or in pgAdmin: right-click Databases -> Create -> fanet_guard)
3. Set the `PGPASSWORD` environment variable if your PostgreSQL user requires a password. Edit `DB_CONFIG` at the top of `fanet_guard.py` to change the host, database, or user.
4. Install libraries: `pip install -r Requirement.txt`

## Run
    python fanet_guard.py train                 # trains model, creates model.pkl
    python fanet_guard.py run                   # live monitoring
    python fanet_guard.py run --simulate-crash  # DEMO: DB "crashes" at tick 8, recovers at 18

## Demo script
1. Run `run --simulate-crash`.
2. Show ALERT lines (NETWORK_DROP / JAMMING) appearing for random drones.
3. Show `[DB DOWN]` lines and `backup_data.csv` growing.
4. Show `recovered N rows from backup file` once the DB is back -> zero data loss.
5. Verify in PostgreSQL:
       SELECT drone_id, status, COUNT(*) FROM telemetry GROUP BY drone_id, status;
       SELECT * FROM telemetry WHERE status <> 'NORMAL' ORDER BY ts DESC LIMIT 10;

You can also crash it for real: stop the PostgreSQL service while `run` is going, then start it again.

## Features used by the ML model
| Feature | Normal | Network drop | Jamming |
|---|---|---|---|
| RSSI (dBm) | ~ -55 | ~ -90 (very weak) | ~ -52 (looks strong) |
| SNR (dB) | ~ 25 | ~ 8 | ~ 3 (collapses) |
| Packet loss % | ~ 2 | 70-100 | 40-90 |
| Latency (ms) | ~ 40 | 500-1000 | 150-400 |

## Future scope
Real telemetry via MAVLink/UDP, live dashboard (Grafana/Streamlit), LSTM time-series model, anomaly detection for unknown attacks.
