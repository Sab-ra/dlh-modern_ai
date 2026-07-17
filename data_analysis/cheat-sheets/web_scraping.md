# Web Scraping Data Collection — Cheat Sheet

> Real life wants robust code.  
> School checker wants exact strings, function names, and emotional support.

---

## 0) Fetch HTML (`requests`)

### Minimal pattern
- `requests.get(url, headers=headers, timeout=timeout)`
- `response.raise_for_status()`  ✅ required for HTTP >= 400
- `return response.text`

### Common fail
- Forgetting `raise_for_status()`.

---

## 1) Basic scraping (`BeautifulSoup`)

### Typical extraction pattern
- Loop cards: `soup.find_all("div", class_="quote")`
- Text: `quote.find("span", class_="text")`
- Author: `quote.find("small", class_="author")`
- Tags: `quote.select("div.tags a.tag")`

### Output shape
Return:
- list of dicts with keys:
  - `"text"` (str)
  - `"author"` (str)
  - `"tags"` (list[str])

### Checker trap
- Import line must often be exact:
  `fetch_html = __import__('0-fetch_html').fetch_html`
- Sometimes checker wants it at **module level**, not inside function.

---

## 2) Pagination

### Core idea
- Start from `base_url`
- Parse current page
- Find `li.next a[href]`
- Build next URL with `urllib.parse.urljoin(...)`
- Stop when no next page

### Polite scraping
- `time.sleep(1)` between pages when required.

---

## 3) API scraping

### Core idea
- Hit `/api/...` endpoint page by page
- Parse JSON
- Read list payload (`quotes`, `items`, etc.)
- Stop on `has_next == False` (or equivalent)

### Common fail
- Wrong author shape handling (`dict` vs `str`).

---

## 4) JSON-LD extraction

### Core idea
- Find:
  `soup.find_all("script", type="application/ld+json")`
- `json.loads(script content)` safely
- Accept both dict and list payloads
- Filter by `@type == "Quote"` (or type list containing `"Quote"`)

### Tag normalization
- `keywords` may be:
  - comma-separated string
  - list
  - missing

---

## 5) Login + scrape (session + CSRF)

### Core flow
1. `GET login page`
2. Extract CSRF token
3. `POST` credentials + token
4. `GET` protected page
5. Parse data

### Checker trap
- Some tests expect unsafe behavior on missing token
  (`token_input["value"]` crash), not safe fallback.
- Yes, this is dumb. Yes, it happens.

---

## 6) Selenium list scraping (static test site)

### Often required imports only
- `import time`
- `from selenium import webdriver`

### Stable extraction for products
From each `.thumbnail`, extract:
- title (`a.title`)
- price (`h4.price`)
- description (`p.description`)
- rating (star count / rating attribute)

### Output shape (strict)
```python
{
  "title": str,
  "price": str,
  "description": str,
  "rating": int
}
```

### Common fail
- Wrong function name expected by checker
  (e.g., `scrape_products`, `scrape_products_list`).
- Missing `return products`.

---

## 7) Selenium single product detail

### Core idea
- Open product URL
- small wait (`time.sleep(delay)`)
- extract title/price/description/rating
- return one dict

### Rating fail pattern
- Selectors vary (`data-rating`, `.glyphicon-star`, `.ws-icon-star`)
- Keep fallback logic.

---

## 8) Infinite scroll + dedupe

### Core idea
- Scroll loop with hard limits:
  - max time
  - max iterations
  - stop when card count stabilizes
- Extract all cards
- Remove duplicates

### Dedupe rule (task-specific)
- Track only:
  `key = (title, price)`
- Not full dict, unless task explicitly says so.

### Timeout prevention
- Avoid endless scroll loops.
- Keep bounded waits.

---

## Fast Debug Checklist

### When terminal works but notebook fails
- Different kernel/interpreter/env/proxy.
- Compare:
  - `sys.executable`
  - `requests.__version__`
  - `HTTP_PROXY` / `HTTPS_PROXY`

### Show true exception
Use `repr(e)`, not silent/pretty printing.

### Verify style quickly
```bash
pycodestyle /workspace/data_analysis/web_scraping/*.py
```

---

## Selector Micro-Recipes

### Quotes
- Card: `div.quote`
- Text: `span.text`
- Author: `small.author`
- Tags: `div.tags a.tag`

### Products (webscraper.io)
- Card: `div.thumbnail`
- Title: `a.title`
- Price: `h4.price`
- Description: `p.description`
- Rating: stars in `.ratings ...`

---

## Survival Rules (House edition)

1. Checker is not reality. It is a ritual.
2. Match required function names exactly.
3. Match required output keys exactly.
4. Keep imports exactly as requested.
5. Keep lines <= 79 chars (`pycodestyle`).
6. First make checker green, then make code beautiful.