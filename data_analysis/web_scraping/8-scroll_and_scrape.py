#!/usr/bin/env python3
"""
Scroll an infinite page and scrape unique product cards with Selenium.
"""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=1.0):
    """
    Scroll with hard stop, then return unique product records.
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

        start_time = time.time()
        max_wait = 22
        max_scrolls = 80
        stable_rounds = 0
        last_height = driver.execute_script(
            "return document.body.scrollHeight"
        )
        last_count = 0

        for _ in range(max_scrolls):
            if time.time() - start_time > max_wait:
                break

            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(scroll_pause)

            new_height = driver.execute_script(
                "return document.body.scrollHeight"
            )
            new_count = len(
                driver.find_elements("css selector", "div.thumbnail")
            )

            if new_height == last_height and new_count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0

            last_height = new_height
            last_count = new_count

            if stable_rounds >= 3:
                break

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

            rating_el = card.find_elements("css selector", "[data-rating]")
            if rating_el:
                raw = rating_el[0].get_attribute("data-rating") or "0"
                if raw.isdigit():
                    rating = int(raw)

            if rating == 0:
                stars = card.find_elements(
                    "css selector",
                    ".ratings .ws-icon-star, .ratings .glyphicon-star",
                )
                rating = len(stars)

            item = {
                "title": str(title),
                "price": str(price),
                "description": str(description),
                "rating": int(rating),
            }

            key = (
                item["title"],
                item["price"],
                item["description"],
                item["rating"],
            )
            if key in seen:
                continue
            seen.add(key)
            products.append(item)

        return products
    finally:
        driver.quit()
