import pandas as pd
import requests
import io
import time

def get_better_mta_data():
    print("Connecting to the api")
    all_month = []
    url = "https://data.ny.gov/resource/wujg-7c2s.csv"
    for i in range(1, 13):
        if i < 9:
            from_time = f"2023-0{i}-01T00:00:00"
            to_time = f"2023-0{i + 1}-01T00:00:00"
        elif i == 9:
            from_time = f"2023-0{i}-01T00:00:00"
            to_time = f"2023-10-01T00:00:00"
        elif i == 10 or i == 11:
            from_time = f"2023-{i}-01T00:00:00"
            to_time = f"2023-{i + 1}-01T00:00:00"
        else:
            from_time = f"2023-{i}-01T00:00:00"
            to_time = f"2024-01-01T00:00:00"

        params = {
            "$where": f"transit_timestamp >= '{from_time}' AND transit_timestamp < '{to_time}'",
            "$limit": 5000000 
        }
        print(f"Downloading Month {i}...", end=" ")

        try:
            response = requests.get(url, params=params)

            if response.status_code == 200:
                df_month = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                all_month.append(df_month)
                print(f"successfully appended {i} month with {len(df_month)} rows")
            else:
                print(f"failed: {response.status_code}")
        except Exception as e:
            print(f"error as {e}")
        time.sleep(1)

    print("Combining monthly file to a single 2023 file")
    full_year_df = pd.concat(all_month, ignore_index = True)
    filename = f"mta_data_2023.csv"
    full_year_df.to_csv(filename, index=False)

if __name__ == "__main__":
    get_better_mta_data()