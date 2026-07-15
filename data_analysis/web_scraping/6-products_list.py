#!/usr/bin/env python3
"""
Fetch products from a JS-rendered page with Selenium.
"""
import time
from selenium import webdriver


def products_list(url):
    """
    Open the page, wait for product cards, and return product data.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        selector = ".product, .product-card, .card"
        cards = []
        start = time.time()
        while time.time() - start < 10:
            cards = driver.find_elements("css selector", selector)
            if cards:
                break
            time.sleep(0.25)

        items = []
        for card in cards:
            title = ""
            price = ""
            link = ""

            title_el = card.find_elements(
                "css selector", "h2, h3, .title, .product-title"
            )
            if title_el:
                title = title_el[0].text.strip()

            price_el = card.find_elements(
                "css selector", ".price, .product-price, [data-price]"
            )
            if price_el:
                price = price_el[0].text.strip()

            link_el = card.find_elements("css selector", "a")
            if link_el:
                link = link_el[0].get_attribute("href") or ""

            items.append(
                {
                    "title": title,
                    "price": price,
                    "url": link,
                }
            )

        return items
    finally:
        driver.quit()