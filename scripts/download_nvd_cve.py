import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error
import datetime as dt

BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
OUT_DIR = "/home/ivancito/dataset/ultimate-cybersecurity-dataset/data/bronze_raw/nvd_cve"

START_DATE = os.getenv("NVD_START_DATE", "2024-01-01")
END_DATE = os.getenv("NVD_END_DATE", dt.date.today().isoformat())
API_KEY = os.getenv("NVD_API_KEY")

os.makedirs(OUT_DIR, exist_ok=True)

combined_path = os.path.join(OUT_DIR, "nvd_cves_2024_to_present.jsonl")

if os.path.exists(combined_path):
    os.remove(combined_path)

def parse_date(value):
    return dt.datetime.strptime(value, "%Y-%m-%d").date()

def month_chunks(start_date, end_date):
    current = start_date
    while current <= end_date:
        next_month = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        chunk_end = min(next_month - dt.timedelta(days=1), end_date)
        yield current, chunk_end
        current = chunk_end + dt.timedelta(days=1)

def fetch_json(params):
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "ultimate-cybersecurity-dataset-research/0.1"
    }

    if API_KEY:
        headers["apiKey"] = API_KEY

    req = urllib.request.Request(url, headers=headers)

    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            wait = 20 + attempt * 20
            print(f"HTTP {e.code}. Waiting {wait}s then retrying...")
            time.sleep(wait)
        except Exception as e:
            wait = 10 + attempt * 10
            print(f"Error: {e}. Waiting {wait}s then retrying...")
            time.sleep(wait)

    raise RuntimeError("Failed after multiple retries.")

start_date = parse_date(START_DATE)
end_date = parse_date(END_DATE)

print(f"Downloading NVD CVEs from {start_date} to {end_date}")
print(f"Output folder: {OUT_DIR}")

with open(combined_path, "a", encoding="utf-8") as combined_file:
    for chunk_start, chunk_end in month_chunks(start_date, end_date):
        print(f"\nPulling {chunk_start} to {chunk_end}")

        start_index = 0
        results_per_page = 2000

        while True:
            params = {
                "pubStartDate": f"{chunk_start}T00:00:00.000",
                "pubEndDate": f"{chunk_end}T23:59:59.999",
                "startIndex": start_index,
                "resultsPerPage": results_per_page
            }

            data = fetch_json(params)

            total_results = int(data.get("totalResults", 0))
            vulnerabilities = data.get("vulnerabilities", [])

            raw_page_name = f"nvd_cves_{chunk_start}_{chunk_end}_start_{start_index}.json"
            raw_page_path = os.path.join(OUT_DIR, raw_page_name)

            with open(raw_page_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            for item in vulnerabilities:
                combined_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Saved {len(vulnerabilities)} records | startIndex={start_index} | total={total_results}")

            if not vulnerabilities or start_index + results_per_page >= total_results:
                break

            start_index += results_per_page

            time.sleep(0.8 if API_KEY else 6)

        time.sleep(0.8 if API_KEY else 6)

print("\nDone.")
print(f"Combined JSONL saved to: {combined_path}")
