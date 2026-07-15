#!/usr/bin/env python3
"""
Scrape product cards from a JS-rendered page with Selenium.
"""
import time
from selenium import webdriver


def scrape_products(url):
    """
    Open page, wait for products, and return normalized product records.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        js = """
const selectors = ['.product', '.product-card', '.card'];
let cards = [];
for (const s of selectors) {
  cards = Array.from(document.querySelectorAll(s));
  if (cards.length) break;
}
return cards.map((card) => {
  const t = card.querySelector('h2, h3, .title, .product-title');
  const p = card.querySelector('.price, .product-price, [data-price]');
  const a = card.querySelector('a[href]');
  return {
    title: t ? t.textContent.trim() : '',
    price: p ? p.textContent.trim() : '',
    url: a ? a.href : ''
  };
});
"""

        items = []
        start = time.time()
        while time.time() - start < 10:
            items = driver.execute_script(js)
            if items:
                break
            time.sleep(0.25)

        normalized = []
        for item in items:
            normalized.append(
                {
                    "title": str(item.get("title", "")),
                    "price": str(item.get("price", "")),
                    "url": str(item.get("url", "")),
                }
            )

        return normalized
    finally:
        driver.quit()


def products_list(url):
    """
    Keep backward compatibility with older function name.
    """
    return scrape_products(url)
