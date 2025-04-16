from selenium import webdriver
from bs4 import BeautifulSoup
import time

options = webdriver.ChromeOptions()
options.add_argument('--disable-blink-features=AutomationControlled')
driver = webdriver.Chrome(options=options)

base_url = "https://nanjing.esf.fang.com/house-a0271/"
listings = []

# obtain page1
driver.get(base_url)
input("✅ After verification, hit Enter to continue crawl......")

# all pages
for page in range(1, 20):
    if page == 1:
        url = base_url
    else:
        url = f"https://nanjing.esf.fang.com/house-a0271/i3{page}/"

    print(f"crawling page {page} : {url}")
    driver.get(url)
    time.sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    for dl in soup.find_all('dl'):
        text = dl.get_text(separator=' ', strip=True)
        if '㎡' in text and '万' in text:
            listings.append(text)

driver.quit()

# discard repeated items
listings = list(set(listings))

# output
print(f"\n✅ obtain {len(listings)} pieces of information in total：\n")
for i, item in enumerate(listings[:10], 1):
    print(f"{i}. {item}")

with open("housing information_pages.txt", "w", encoding="utf-8") as f:
    for item in listings:
        f.write(item + "\n")

print("\n📄 saved as：housing information_pages.txt")