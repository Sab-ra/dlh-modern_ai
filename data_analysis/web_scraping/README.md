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

