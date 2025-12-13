import sqlite3

con = sqlite3.connect("monitoring.db")
cur = con.cursor()
cur.execute("""
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id INTEGER REFERENCES requests(request_id),
  data_id INTEGER,
  VendorID INTEGER,
  passenger_count INTEGER,
  trip_distance REAL,
  pickup_hour INTEGER,
  pickup_day_of_week INTEGER,
  pickup_month INTEGER,
  pickup_day INTEGER,
  PULocationID INTEGER,
  DOLocationID INTEGER,
  predicted_duration REAL,
  was_checked INTEGER DEFAULT 0
);
""")

cur.execute("""CREATE TABLE requests (request_id INTEGER PRIMARY KEY);""")

cur.execute("""
  CREATE TABLE labels (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id INTEGER REFERENCES requests(request_id),
  label_id INTEGER,
  real_duration REAL
);
""")

con.commit()
cur.close()
con.close()
