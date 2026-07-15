# Web Scraping Data Collection

## 0-fetch_html.py

Write a function `def fetch_html(url, headers = None, timeout = 10):` that fetches a web page and returns its HTML as text:

- [x] url is the page to retrieve
- [x] headers is an optional dict of HTTP headers (e.g. {"User-Agent": "…”})
- [x] timeout is the number of seconds to wait before aborting
- [x] Must raise an exception on any HTTP status ≥ 400
- [x] Returns: the full HTML of the response as a string

_Only import: import requests_

## 1-scrape_basic.py

Write a function `def scrape_basic(url):` that scrapes the first page of quotes from **quotes.toscrape.com**:
- [x] 
- [x] url is the Quotes List endpoint (e.g. https://quotes.toscrape.com/)
- [x] Use fetch_html() to retrieve the HTML then parse it with **BeautifulSoup**
- [x] Extract for each quote block:
    - "text": the quote text
    - "author": the quote’s author
    - "tags": a list of tag strings
- [x] You are not allowed to use regular expressions for this task
- [x] Returns: a list of dicts, e.g. [{ "text": "...", "author": "...", "tags": [...] }, …]

_Imports: `from bs4 import BeautifulSoup` and `fetch_html = __import__('0-fetch_html').fetch_html`_

## 2-scrape_paginated.py

Write a function `def scrape_paginated(base_url):` that follows “Next” links on **quotes.toscrape.com** until no more pages remain:

- [ ] base_url is the first page URL (https://quotes.toscrape.com/)
- [ ] Must detect and follow the <li class="next"><a href="…"> link dynamically
- [ ] Implement delays between requests (e.g. time.sleep)
- [ ] Combine results from all pages into one list
- [ ] Returns: the full list of quote dicts (same format as Task 1)

_Imports: `from bs4 import BeautifulSoup`, `import time`, `from urllib import parse`, `fetch_html = __import__('0-fetch_html').fetch_html and scrape_basic = __import__('1-scrape_basic').scrape_basic`_

## 3-scrape_via_api.py

Write a function def scrape_via_api(base_url): that fetches quote data from all the quotes' API pages:

- [x] base_url is the root URL of the site (e.g. "https://quotes.toscrape.com")
- [x] Build each page’s API endpoint starting from page one (/api/quotes?page=1)
- [x] Use fetch_html() to retrieve the JSON payload
- [x] From each page’s "quotes" array, extract:
    - "text": the quote text
    - "author": the author’s name
    - "tags": the list of tags
- [x] Return: a list of quote dicts, each with keys "text", "author", and "tags"

_Imports: `import json` and `fetch_html = __import__('0-fetch_html').fetch_html`_

## 4-extract_jsonld.py

Write a function `def extract_jsonld(url):` that pulls quotes from embedded JSON‑LD on a page:

- [x] url is the Quotes List endpoint (e.g. "https://quotes.toscrape.com/")
- [x] Use fetch_html() to fetch the HTML
- [x] Locate all `<script type="application/ld+json">` blocks and parse each with `json.loads()`
- [x] From each JSON‑LD object of "@type": "Quote", extract:
- [x] "text": the quote text (.get("text"))
- [x] "author": the author’s name (.get("author", {}).get("name"))
- [x] "tags": keywords, (p.s. split into a list if provided as a comma-separated string)
- [x] Return: a list of quote dicts, each with keys "text", "author", and "tags"

_Imports: `import json`, `from bs4 import BeautifulSoup` and `fetch_html = __import__('0-fetch_html').fetch_html`_

## 5-login_and_scrape.py

Write a function `def login_and_scrape(login_url, user, pwd):` that logs in and scrapes quotes visible only after authentication:

- [x] login_url is the login page (e.g. "https://quotes.toscrape.com/login")
- [x] Use requests.Session() to persist cookies across requests
- [x] GET the login form and extract the CSRF token from <input name="csrf_token">
- [x] POST credential fields (username, password, csrf_token) back to login_url
- [x] After successful login, GET the protected quotes page (https://quotes.toscrape.com/)
- [x] Use BeautifulSoup to parse each `<div class="quote">` and extract:
    - "text": the quote text
    - "author": the author’s name
    - "tags": a list of tag strings
- [x] Return: a list of quote dicts, each with keys "text", "author", and "tags"

_Imports: `import requests`, `from bs4 import BeautifulSoup`_

## 6-products_list.py

Write a function `def scrape_products_list(url):` that opens a static product category page and returns a list of product dictionaries. Each dict should have:

- [x]"title": the product’s name (from the title attribute of the <a> tag)
- [x]"price": the product’s price (text of the <h4 class="price"> element)
- [x]"description": the product’s description (text of the <p class="description">)
- [x]"rating": the number of stars (<p data-rating="rating_value"> under .ratings)

_Imports: `import time`, `from selenium import webdriver`_

### For this task 6 and the rest of the tasks:

- Use only Selenium (webdriver, By, etc.)
- Run Chrome in headless mode in a 1920 by 1080 window and no sandbox.
- Don’t use BeautifulSoup or regex

## 7-product_detail.py

**Scrape Single Product Detail**

Write a function `def scrape_product_detail(url, delay=2.0):` that opens a detail page for one product, waits delay seconds, and returns a dictionary with:

- [ ]"title": the product title (the second <h4> inside .caption)
- [ ]"price": the price (text of the first <h4 class="price">)
- [ ]"description": the full description (text of <p class="description">)
- [ ]"rating": the number of stars (count of <p class="ws-icon ws-icon-star"> in .ratings)

**Use only Selenium with find_element(s), CSS selectors, and no external parsers.**

_Imports: `import time`, `from selenium import webdriver`_
