import os
import json
import re
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
import time # Import time for delays

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34 # Adjust if your category ID is different
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
AUTH_STATE_FILE = "auth.json"
DATE_FROM = datetime(2025, 1, 1) # Adjust your start date
DATE_TO = datetime(2025, 4, 14) # Adjust your end date (or datetime.now() if you want current)

# Output directory for individual topic JSON files
DISCOURSE_OUTPUT_DIR = "downloaded_threads"
os.makedirs(DISCOURSE_OUTPUT_DIR, exist_ok=True)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

def login_and_save_auth(playwright):
    print("🔐 No auth found. Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False) # Open visible browser
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{BASE_URL}/login")
    print("🌐 Please log in manually using Google in the browser window. Then press ▶️ (Resume) in Playwright bar.")
    page.pause() # This will pause script execution until you manually resume it in the browser
    context.storage_state(path=AUTH_STATE_FILE)
    print("✅ Login state saved.")
    browser.close()

def is_authenticated(page):
    try:
        # Try to load a JSON page to check authentication status
        page.goto(CATEGORY_JSON_URL, timeout=10000)
        # Check if the page content is likely JSON (starts with { and ends with })
        page.wait_for_function('document.body.innerText.trim().startsWith("{") && document.body.innerText.trim().endsWith("}")', timeout=5000)
        json.loads(page.inner_text("body")) # Try to parse body as JSON
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False

def scrape_posts(playwright):
    print("🔍 Starting Discourse scrape using saved session...")
    browser = playwright.chromium.launch(headless=True) # Run headless for actual scraping
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()

    all_topics_metadata = [] # This will store simplified topic metadata from category pages
    page_num = 0
    while True:
        paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
        print(f"📦 Fetching Discourse category page {page_num}...")
        try:
            page.goto(paginated_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_function('document.body.innerText.trim().startsWith("{") && document.body.innerText.trim().endsWith("}")', timeout=10000)
            data = json.loads(page.inner_text("body")) # Read from body in case <pre> is not used
        except (TimeoutError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load or parse category page {page_num}: {e}. Assuming no more pages.")
            break

        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break # No more topics means end of pagination

        all_topics_metadata.extend(topics)
        page_num += 1
        time.sleep(0.5) # Be polite between category page requests

    print(f"📄 Found {len(all_topics_metadata)} total topics across all Discourse category pages.")

    downloaded_topic_count = 0
    for topic_meta in all_topics_metadata:
        created_at = parse_date(topic_meta["created_at"])
        if DATE_FROM <= created_at <= DATE_TO:
            topic_id = topic_meta['id']
            topic_slug = topic_meta['slug']
            topic_url = f"{BASE_URL}/t/{topic_slug}/{topic_id}.json"
            
            output_file_path = os.path.join(DISCOURSE_OUTPUT_DIR, f"topic_{topic_id}.json")

            if os.path.exists(output_file_path):
                print(f"⏩ Topic {topic_id} already downloaded. Skipping.")
                downloaded_topic_count += 1
                continue

            print(f"🔽 Downloading Discourse topic JSON: {topic_url}")
            try:
                page.goto(topic_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_function('document.body.innerText.trim().startsWith("{") && document.body.innerText.trim().endsWith("}")', timeout=10000)
                topic_data = json.loads(page.inner_text("body")) # This is the full topic data
                
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(topic_data, f, indent=2)
                print(f"✅ Saved full topic {topic_id} to {output_file_path}")
                downloaded_topic_count += 1
                time.sleep(0.5) # Be polite: add a small delay between topic requests
            except (TimeoutError, json.JSONDecodeError) as e:
                print(f"❌ Error downloading or parsing topic {topic_id} from {topic_url}: {e}")
            
    print(f"✅ Finished Discourse scraping. Downloaded {downloaded_topic_count} topics within the date range to '{DISCOURSE_OUTPUT_DIR}'.")
    browser.close()

def main():
    with sync_playwright() as p:
        if not os.path.exists(AUTH_STATE_FILE):
            login_and_save_auth(p)
        else:
            browser_check = p.chromium.launch(headless=True)
            context_check = browser_check.new_context(storage_state=AUTH_STATE_FILE)
            page_check = context_check.new_page()
            if not is_authenticated(page_check):
                print("⚠️ Session invalid. Re-authenticating...")
                browser_check.close()
                login_and_save_auth(p)
            else:
                print("✅ Using existing authenticated session.")
                browser_check.close() # Close the check browser before the main scrape starts

        scrape_posts(p)

if __name__ == "__main__":
    main()