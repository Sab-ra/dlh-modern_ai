#!/usr/bin/env python3
"""
Scroll an infinite page and scrape unique product cards with Selenium.
"""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """
    Scroll until page height stops growing, then scrape unique products.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )
        stable_rounds = 0

        while stable_rounds < 2:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(scroll_pause)

            new_height = driver.execute_script(
                "return document.body.scrollHeight"
            )
            if new_height == last_height:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_height = new_height

        cards = driver.find_elements("css selector", "div.thumbnail")
        products = []
        seen = set()

        for card in cards:
            title = ""
            price = ""
            description = ""
            rating = 0

            title_el = card.find_elements("css selector", "a.title")
            if title_el:
                title = title_el[0].get_attribute("title") or ""
                if not title:
                    title = title_el[0].text.strip()

            price_el = card.find_elements("css selector", "h4.price")
            if price_el:
                price = price_el[0].text.strip()

            desc_el = card.find_elements("css selector", "p.description")
            if desc_el:
                description = desc_el[0].text.strip()

            stars = card.find_elements(
                "css selector",
                ".ratings p.ws-icon.ws-icon-star",
            )
            if not stars:
                stars = card.find_elements(
                    "css selector",
                    ".ratings .ws-icon-star",
                )
            rating = len(stars)

            if rating == 0:
                rating_el = card.find_elements(
                    "css selector",
                    ".ratings [data-rating]",
                )
                if rating_el:
                    raw = rating_el[0].get_attribute("data-rating") or "0"
                    if raw.isdigit():
                        rating = int(raw)

            key = (title, price)
            if key in seen:
                continue
            seen.add(key)

            products.append(
                {
                    "title": str(title),
                    "price": str(price),
                    "description": str(description),
                    "rating": int(rating),
                }
            )

        return products
    finally:
        driver.quit()
