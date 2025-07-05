import json
import os
import io
from typing import Optional
import requests
from playwright.sync_api import sync_playwright

from settings import custom_logger, load_settings
from structs.product import Product
from structs.storage_type import StorageType
from connectors.s3_client import S3Client

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

class ProductScraper:
    def __init__(self, output_dir: str = "data/scraped_data") -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.output_dir = output_dir
        self.images_dir = os.path.join(self.output_dir, "images")
        self.products_dir = os.path.join(self.output_dir, "products")

        # Configuración de almacenamiento
        storage_config = load_settings("Storage")
        self.storage_type = StorageType(storage_config["Type"])
        if self.storage_type == StorageType.S3:
            self.s3_client = S3Client(
                bucket_name=storage_config["S3"]["Bucket"],
                region_name=storage_config["S3"]["Region"]
            )
        else:
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.products_dir, exist_ok=True)

        # Configuración de scraping
        config = load_settings("WebPage")
        self.base_url = config["BaseUrl"]
        self.validation_url = config["ValidationUrl"]
        self.max_products = config["MaxProducts"]

    def run(self) -> None:
        self.logger.info("Starting product scraper")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.base_url)

            # Scroll para cargar más productos
            for _ in range(100):
                page.mouse.wheel(0, 5000)
                page.wait_for_timeout(1000)

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
                name_el = element.query_selector('.vtex-product-summary-2-x-productBrand')
                name = name_el.inner_text().strip() if name_el else "N/A"

                price_el = element.query_selector('.devotouy-products-components-0-x-sellingPriceWithUnitMultiplier span:last-child')
                price = 0.0
                if price_el:
                    price_text = price_el.inner_text().replace('.', '').replace(',', '.').strip()
                    try:
                        price = float(price_text)
                    except Exception:
                        price = 0.0

                brand_el = element.query_selector('.vtex-product-summary-2-x-productBrandName')
                brand = brand_el.inner_text().strip() if brand_el else None

                img_el = element.query_selector('img.vtex-product-summary-2-x-image')
                img_url = img_el.get_attribute('src') if img_el else None
                images = [img_url] if img_url else []

                link_el = element.query_selector('a.vtex-product-summary-2-x-clearLink')
                link = link_el.get_attribute('href') if link_el else None
                if link and not link.startswith("http"):
                    link = self.validation_url.rstrip("/") + link

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

    def _save_product(self, product: Product) -> None:
        product_img_dir = os.path.join("data", "scraped_data", "images", product.id)
        os.makedirs(product_img_dir, exist_ok=True)

        for img_url in product.images:
            if img_url:
                img_filename = img_url.split("/")[-1].split("?")[0]
                _, ext = os.path.splitext(img_filename)
                if ext.lower() not in VALID_IMAGE_EXTENSIONS:
                    ext = ".jpg"
                    img_filename = f"{product.id}{ext}"
                img_path = os.path.join("data", "scraped_data", "images", product.id, img_filename)

                try:
                    img_data = requests.get(img_url).content
                    if self.storage_type == StorageType.S3:
                        img_file = io.BytesIO(img_data)
                        self.logger.info(f"Subiendo imagen con key: {img_path}")
                        self.s3_client.upload_image(
                            file_obj=img_file,
                            key=img_path,
                            content_type="image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
                        )
                    else:
                        with open(img_path, "wb") as f:
                            f.write(img_data)
                        self.logger.info(f"Imagen guardada en {img_path}")
                except Exception as e:
                    self.logger.error(f"Failed to download image {img_url}: {str(e)}")

        jsonl_path = os.path.join("data", "scraped_data", "products", f"{product.id}.jsonl")

        if self.storage_type == StorageType.S3:
            self.logger.info(f"Subiendo producto con key: {jsonl_path}")
            self.s3_client.save_jsonl([product.dict()], key=jsonl_path)
        else:
            try:
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(product.dict(), ensure_ascii=False) + "\n")
                self.logger.info(f"Saved product {product.id}")
            except Exception as e:
                self.logger.error(f"Failed to save product {product.id}: {str(e)}")