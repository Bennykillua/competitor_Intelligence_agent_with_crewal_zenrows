
import os
import requests
from dotenv import load_dotenv

load_dotenv()

ZENROWS_API_KEY = os.environ["ZENROWS_API_KEY"]

# Same protected competitor URL used with ScrapeWebsiteTool
url = "https://www.yelp.com"

params = {
    "url": url,
    "apikey": ZENROWS_API_KEY,
    "js_render": "true",
    "premium_proxy": "true"
}

response = requests.get(
    "https://api.zenrows.com/v1/",
    params=params,
    timeout=180
)

print("status code:", response.status_code)
print("bytes returned:", len(response.text))
print(response.text[:1000])