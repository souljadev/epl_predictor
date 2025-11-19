import http.client
import json
import pandas as pd

# --- API CALL ---
conn = http.client.HTTPSConnection("v3.football.api-sports.io")

headers = {
    'x-apisports-key': "5aad12c91bfb128edb45281d39a03a39"
}

conn.request("GET", "/odds?league=39&season=2025", headers=headers)
res = conn.getresponse()
print("Status Code:", res.status)

data = res.read().decode("utf-8")
print(data)
# --- PARSE JSON ---
json_data = json.loads(data)

# The data is nested under "response" according to API-Sports structure
records = json_data.get("response", [])

# --- NORMALIZE JSON INTO TABLE ---
df = pd.json_normalize(records)

# --- EXPORT TO CSV ---
output_path = "data/raw/odds.csv"
df.to_csv(output_path, index=False)

print(f"CSV created: {output_path}")
print(df.head())
