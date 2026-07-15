#!/usr/bin/env python3
"""
Scroll a dynamic products page and return unique product records.
"""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=0.4):
    """
    Scroll until product count stabilizes, then scrape unique products.
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

        max_wait = 20.0
        max_steps = 70
        stable_rounds = 0
        last_count = 0
        start_time = time.time()

        for _ in range(max_steps):
            if time.time() - start_time > max_wait:
                break

            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(scroll_pause)

            cards = driver.find_elements("css selector", "div.thumbnail")
            current_count = len(cards)

            if current_count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
                last_count = current_count

            if stable_rounds >= 10:
                break

        time.sleep(0.6)
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
                ".ratings .ws-icon-star, .ratings .glyphicon-star",
            )
            rating = len(stars)

            key = (str(title), str(price))
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
