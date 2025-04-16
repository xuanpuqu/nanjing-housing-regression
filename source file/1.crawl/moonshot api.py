import requests
import pandas as pd
import time
import json
import re

# ===== 1. Read text from file =====
with open("housing information_pages.txt", "r", encoding="utf-8") as f:
    listings = [line.strip() for line in f if line.strip()]

print(f" Loaded {len(listings)} housing text entries")

# ===== 2. Moonshot API configuration =====
MOONSHOT_API_KEY = "sk-TFUxae4TAAl35wOivH7dZHgHQTtchcf602lGQfPZTQJBlRK0"
API_URL = "https://api.moonshot.cn/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {MOONSHOT_API_KEY}",
    "Content-Type": "application/json",
}

# ===== 3. Function to extract structured data using Moonshot =====
def extract_info_moonshot(texts):
    results = []
    for i, text in enumerate(texts):
        print(f"⏳ Entry {i+1}: {text[:30]}...", flush=True)
        prompt = f"""
Please extract the following structured information from the housing description below, and return in JSON format:
- Layout (e.g. 2 bedrooms, 1 living room)
- Area (㎡)
- Unit price (CNY/㎡)
- Orientation
- Floor (e.g. middle floor)
- Total floors (e.g. 18 floors)
- Decoration (e.g. simple, refined, rough)
- Compound/Community name
- Transportation (e.g. metro, bus)
- District (e.g. Qixia, Yuhuatai)
- Remarks (one-sentence summary)

Original housing description:
{text}

Please output only JSON in the following format:
{{
  "户型": "", "建筑面积": "", "单价": "", "朝向": "", "楼层": "",
  "总楼层": "", "装修": "", "小区": "", "交通": "", "区域": "", "备注": ""
}}
"""

        try:
            res = requests.post(API_URL, headers=headers, json={
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }, timeout=30)

            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                try:
                    # Remove markdown formatting if present
                    content = content.replace("```json", "").replace("```", "").strip()
                    match = re.search(r"\{[\s\S]*\}", content)
                    if match:
                        json_str = match.group(0)
                        result = json.loads(json_str)
                        results.append(result)
                    else:
                        print(f"⚠️ Entry {i+1}: JSON block not found")
                        results.append({"Parse failed": content})
                except Exception as e:
                    print(f"⚠️ Entry {i+1} parsing error: {e}")
                    results.append({"Parse failed": content})
            else:
                print(f"❌ Entry {i+1} API error: {res.status_code}")
                print(res.text)
                results.append({"Request failed": f"Status code: {res.status_code}"})
        except Exception as e:
            print(f" Entry {i+1} exception: {e}")
            results.append({"Exception": str(e)})

        time.sleep(21)  # Prevent rate-limiting by spacing out API calls

    return pd.json_normalize(results)

# ===== 4. Execute extraction (recommend test on first 5 entries) =====
df = extract_info_moonshot(listings[:5])

# ===== 5. Save result to Excel =====
df.to_excel("structured_housing_info_moonshot.xlsx", index=False)
print(" Extraction complete. Saved to: structured_housing_info_moonshot.xlsx")