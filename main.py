from selenium import webdriver


def main() -> None:
    driver = webdriver.Safari()
    try:
        driver.get("https://www.google.com")
        print(driver.title)
        input("Press Enter to close...")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
