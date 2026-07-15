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

