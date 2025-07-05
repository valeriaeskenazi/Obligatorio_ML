import json
import os
import sys
import yaml

sys.path.append(os.getcwd())

from playwright.sync_api import sync_playwright
import requests

from src.settings import custom_logger
from src.structs.product import Product

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'settings', 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['WebPage']

class ProductScraper:
    def __init__(self, output_dir: str = "data/scraped_data") -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.products_dir = os.path.join(self.output_dir, "products")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.products_dir, exist_ok=True)

        # Cargar configuración SOLO desde config.yml
        config = load_config()
        self.base_url = config["BaseUrl"]
        self.validation_url = config["ValidationUrl"]
        self.max_products = config["AmountOfProperties"]

    def run(self) -> None:
        self.logger.info("Starting product scraper")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.base_url)
            products = self._get_products(page)
            self.logger.info(f"Found {len(products)} products")
            for product in products[:self.max_products]:
                self._save_product(product)
            browser.close()
        self.logger.info("Finished scraping products")

    def _get_products(self, page):
        product_elements = page.query_selector_all('.devotouy-search-result-3-x-galleryItem')
        products = []
        for element in product_elements:
            try:
                # Nombre
                name_el = element.query_selector('.vtex-product-summary-2-x-productBrand')
                name = name_el.inner_text().strip() if name_el else "N/A"

                # Precio
                price_el = element.query_selector('.devotouy-products-components-0-x-sellingPriceWithUnitMultiplier span:last-child')
                price = 0.0
                if price_el:
                    price_text = price_el.inner_text().replace('.', '').replace(',', '.').strip()
                    try:
                        price = float(price_text)
                    except Exception:
                        price = 0.0

                # Marca
                brand_el = element.query_selector('.vtex-product-summary-2-x-productBrandName')
                brand = brand_el.inner_text().strip() if brand_el else None

                # Imagen
                img_el = element.query_selector('img.vtex-product-summary-2-x-image')
                img_url = img_el.get_attribute('src') if img_el else None
                images = [img_url] if img_url else []

                # Link
                link_el = element.query_selector('a.vtex-product-summary-2-x-clearLink')
                link = link_el.get_attribute('href') if link_el else None
                if link and not link.startswith("http"):
                    link = self.validation_url.rstrip("/") + link

                # ID
                prod_id = link.split('-')[-1].replace('/p', '') if link else name.replace(' ', '_')

                product = Product(
                    id=prod_id,
                    name=name,
                    link=link,
                    price=price,
                    brand=brand,
                    images=images
                )
                products.append(product)
            except Exception as e:
                self.logger.error(f"Error parsing product: {e}")
        return products

    def _save_product(self, product: Product):
        # Guardar imágenes
        product_img_dir = os.path.join(self.images_dir, product.id)
        os.makedirs(product_img_dir, exist_ok=True)
        for img_url in product.images:
            if img_url:
                # Extraer el nombre y extensión real antes de los parámetros
                img_filename = img_url.split("/")[-1].split("?")[0]
                _, ext = os.path.splitext(img_filename)
                # Si no hay extensión válida, usa .jpg por defecto
                if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                    ext = ".jpg"
                    img_filename = img_filename if img_filename else f"{product.id}{ext}"
                    if not img_filename.endswith(ext):
                        img_filename += ext
                img_path = os.path.join(product_img_dir, img_filename)
                try:
                    img_data = requests.get(img_url).content
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    self.logger.info(f"Imagen guardada en {img_path}")
                except Exception as e:
                    self.logger.error(f"Failed to download image {img_url}: {str(e)}")
        # Guardar producto en JSONL
        jsonl_path = os.path.join(self.products_dir, f"{product.id}.jsonl")
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(product.dict(), ensure_ascii=False) + "\n")
            self.logger.info(f"Saved product {product.id}")
        except Exception as e:
            self.logger.error(f"Failed to save product {product.id}: {str(e)}")