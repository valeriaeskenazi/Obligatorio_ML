from scrapers.disco import ProductScraper
from settings.settings import load_settings
from settings.logger import custom_logger

logger = custom_logger("Main")

if __name__ == "__main__":
    # Inicializar el scraper de productos
    scraper = ProductScraper(output_dir="data/scraped_data")
    scraper.run()