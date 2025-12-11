import pandas as pd
import requests
import time
import random
import argparse
import sys
from datetime import datetime

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration
FLASK_URL = "http://localhost:8080/predict"
DATA_PATH = "data/raw/yellow_tripdata_2011-01.parquet"  # Changed to a reliable file

def load_data(path):
    print(f"Loading data from {path}...")
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def prepare_payload(row):
    # Convert row to dictionary and handle datetime serialization
    data = row.to_dict()
    
    # Convert ALL timestamp objects to strings (not just pickup_datetime)
    # And handle NaN values for JSON compliance
    for key, value in data.items():
        if isinstance(value, (pd.Timestamp, datetime)):
            data[key] = str(value)
        elif pd.isna(value):  # Check for NaN/NaT
            data[key] = None
    
    # Construct payload expected by the server
    payload = {
        "data": [data]
    }
    return payload

def get_drift_cluster(df):
    """
    Uses K-Means to find a cluster of data that represents 'extreme' behavior
    (e.g., long trips) to simulate drift.
    """
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not found. Falling back to simple filtering.")
        return df[df['trip_distance'] > 10]

    print("Clustering data to find a drift scenario...")
    # Select features for clustering (ensure no NaNs)
    features = ['trip_distance', 'passenger_count']
    
    # Sample if too large to speed up K-Means
    if len(df) > 10000:
        print(f"Dataframe too large ({len(df)} rows). Sampling 10,000 rows for clustering.")
        df_sample = df.sample(10000, random_state=42)
        X = df_sample[features].dropna()
    else:
        X = df[features].dropna()
    
    if len(X) < 50:
        print("Not enough data for clustering. Using simple filtering.")
        return df[df['trip_distance'] > 5]

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into 3 groups (e.g., Short, Medium, Long trips)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df.loc[X.index, 'cluster'] = kmeans.fit_predict(X_scaled)
    
    # Find the cluster with the highest average trip_distance
    cluster_means = df.groupby('cluster')['trip_distance'].mean()
    target_cluster = cluster_means.idxmax()
    
    print(f"Identified Cluster {target_cluster} as the 'High Distance' cluster (Mean distance: {cluster_means[target_cluster]:.2f})")
    
    drift_df = df[df['cluster'] == target_cluster]
    return drift_df

def simulate_normal(df, duration_sec=60):
    print(f"Starting NORMAL simulation for {duration_sec} seconds...")
    end_time = time.time() + duration_sec
    
    while time.time() < end_time:
        # Pick a random row
        row = df.sample(1).iloc[0]
        payload = prepare_payload(row)
        
        try:
            resp = requests.post(FLASK_URL, json=payload)
            print(f"Status: {resp.status_code}, Prediction: {resp.json().get('predictions', 'Error')}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(0.5) # 2 requests per second

def simulate_drift(df, duration_sec=300):
    print(f"Starting DRIFT simulation for {duration_sec} seconds...")
    
    # Use clustering to find a drift dataset
    drift_df = get_drift_cluster(df)
    
    if drift_df.empty:
        print("No drift data found. Artificially creating it.")
        drift_df = df.sample(50).copy()
        drift_df['trip_distance'] = drift_df['trip_distance'] + 50.0 # Add 50 miles
    
    print(f"Simulating using {len(drift_df)} samples from the drift cluster...")
    
    end_time = time.time() + duration_sec
    
    while time.time() < end_time:
        row = drift_df.sample(1).iloc[0].copy()
        
        # Force extreme drift to ensure alert triggers (Trip Distance > 50)
        # The alert requires > 20% of trips to be > 50 miles.
        if random.random() < 0.6: # 60% of traffic will be extreme
            row['trip_distance'] = random.uniform(55.0, 120.0)

        payload = prepare_payload(row)
        
        try:
            resp = requests.post(FLASK_URL, json=payload)
            print(f"Status: {resp.status_code}, Prediction: {resp.json().get('predictions', 'Error')}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(0.2) # 5 requests per second to fill the bucket faster

def simulate_random_subset(df, duration_sec=60):
    print(f"Starting RANDOM SUBSET simulation for {duration_sec} seconds...")
    # Pick a random 20% subset of the data and replay it
    subset_size = int(len(df) * 0.2)
    subset_df = df.sample(subset_size)
    print(f"Selected random subset of {subset_size} samples.")
    
    end_time = time.time() + duration_sec
    
    while time.time() < end_time:
        row = subset_df.sample(1).iloc[0]
        payload = prepare_payload(row)
        
        try:
            resp = requests.post(FLASK_URL, json=payload)
            print(f"Status: {resp.status_code}, Prediction: {resp.json().get('predictions', 'Error')}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(0.5)

def simulate_errors(duration_sec=60):
    print(f"Starting ERROR simulation for {duration_sec} seconds...")
    end_time = time.time() + duration_sec
    
    while time.time() < end_time:
        # Send invalid payload (missing columns)
        payload = {"data": [{"invalid_col": 123}]}
        
        try:
            resp = requests.post(FLASK_URL, json=payload)
            print(f"Status: {resp.status_code} (Expected 400/500)")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate traffic for Drift Detection")
    parser.add_argument("mode", choices=["normal", "drift", "random_subset", "error"], help="Simulation mode")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--file", type=str, default=DATA_PATH, help="Path to parquet file")
    
    args = parser.parse_args()
    
    if args.mode in ["normal", "drift", "random_subset"]:
        df = load_data(args.file)
        if args.mode == "normal":
            simulate_normal(df, args.duration)
        elif args.mode == "drift":
            simulate_drift(df, args.duration)
        elif args.mode == "random_subset":
            simulate_random_subset(df, args.duration)
    elif args.mode == "error":
        simulate_errors(args.duration)
