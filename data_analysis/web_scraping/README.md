# Web Scraping Data Collection

## 0-fetch_html.py

Write a function `def fetch_html(url, headers = None, timeout = 10):` that fetches a web page and returns its HTML as text:

- [ ] url is the page to retrieve
- [ ] headers is an optional dict of HTTP headers (e.g. {"User-Agent": "…”})
- [ ] timeout is the number of seconds to wait before aborting
- [ ] Must raise an exception on any HTTP status ≥ 400
- [ ] Returns: the full HTML of the response as a string

_Only import: import requests_