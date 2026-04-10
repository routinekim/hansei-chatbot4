import requests
from bs4 import BeautifulSoup

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://www.hansei.ac.kr/kor/302/subview.do"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
response = requests.get(url, verify=False, timeout=10, headers=headers)
response.encoding = 'utf-8'
soup = BeautifulSoup(response.text, 'html.parser')

items = soup.select('.calendar_list dl')
print(f"Found {len(items)} items using .calendar_list dl")

if not items:
    # Print the soup size and basic structure to see what we actually got
    print("Failed to find items. Response text length:", len(response.text))
    print("Does it contain '1월' or '학사일정'?", "1월" in response.text, "학사일정" in response.text)
    
    # Save the output to a file for investigation
    with open("scrape_debug.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Saved to scrape_debug.html")

else:
    for item in items[:2]:
        dt = item.find('dt').text.strip() if item.find('dt') else ""
        print("Month:", dt)
        print("Events:", len(item.find_all('li')))
