import requests
import json

url = "https://tds-project-1-dusky.vercel.app/api/"
headers = {"Content-Type": "application/json"}
data = {"question": "What are the common tools used in data science?"}

print(f"Attempting to POST to {url} with data: {data}")
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

    print("\n--- API Response (JSON) ---")
    json_response = response.json()
    print(json.dumps(json_response, indent=2))

    # Check if the expected keys are present
    if "answer" in json_response and "sources" in json_response:
        print("\nAPI response contains 'answer' and 'sources' keys. Structure looks correct.")
    else:
        print("\nWARNING: API response might be missing 'answer' or 'sources' keys.")

except requests.exceptions.ConnectionError as e:
    print(f"\nERROR: Could not connect to the API. Is your FastAPI server running? (Error: {e})")
except requests.exceptions.HTTPError as e:
    print(f"\nERROR: API returned an HTTP error: {e.response.status_code} - {e.response.text}")
except json.JSONDecodeError:
    print(f"\nERROR: API response was not valid JSON. Raw response: {response.text}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")