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

url is the Quotes List endpoint (e.g. https://quotes.toscrape.com/)
Use fetch_html() to retrieve the HTML then parse it with **BeautifulSoup**
Extract for each quote block:
    - "text": the quote text
    - "author": the quote’s author
    - "tags": a list of tag strings
You are not allowed to use regular expressions for this task
Returns: a list of dicts, e.g. [{ "text": "...", "author": "...", "tags": [...] }, …]

_Imports: `from bs4 import BeautifulSoup` and `fetch_html = __import__('0-fetch_html').fetch_html`_

## 2-scrape_paginated.py

