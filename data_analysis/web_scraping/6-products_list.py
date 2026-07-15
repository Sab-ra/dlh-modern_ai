#!/usr/bin/env python3
"""
Fetch a product list from a JS-rendered page using Selenium.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def products_list(url):
    """
    Open a products page, wait for cards, and return structured data.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)

        wait = WebDriverWait(driver, 10)
        wait.until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, ".product, .product-card, .card")
            )
        )

        items = []
        cards = driver.find_elements(
            By.CSS_SELECTOR, ".product, .product-card, .card"
        )

        for card in cards:
            title = ""
            price = ""
            link = ""

            title_el = card.find_elements(
                By.CSS_SELECTOR, "h2, h3, .title, .product-title"
            )
            if title_el:
                title = title_el[0].text.strip()

            price_el = card.find_elements(
                By.CSS_SELECTOR, ".price, .product-price, [data-price]"
            )
            if price_el:
                price = price_el[0].text.strip()

            link_el = card.find_elements(By.CSS_SELECTOR, "a")
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
