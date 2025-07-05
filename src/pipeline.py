from connectors.s3_client import S3Client
from connectors.openai_client import clasificar_octogono
from settings.settings import load_settings
from settings.logger import custom_logger
from utils.io_utils import guardar_csv
from scrapers import disco
import argparse
import os
import boto3

from scrapers.disco import ProductScraper



logger = custom_logger("Pipeline")

def run_scraper():
    # ProductScraper SIEMPRE toma el bucket y región del archivo de configuración
    scraper = ProductScraper()
    scraper.run()

def run_tagger(bucket_name, region_name, prefix, output_file):
    # S3Client personalizado para listar y descargar imágenes desde S3
    s3 = S3Client(bucket_name=bucket_name, region_name=region_name)
    claves = s3.list_files(prefix=prefix)
    resultados = []

    for key in claves:
        try:
            logger.info(f"Procesando imagen: {key}")
            imagen_bytes = s3.download_image(key)
            etiqueta = clasificar_octogono(imagen_bytes)
            resultados.append((key, etiqueta))
        except Exception as e:
            logger.error(f"❌ Error procesando {key}: {e}")

    guardar_csv(resultados, output_file)
    logger.info(f"\n✅ Proceso terminado. Resultados guardados en {output_file}")
    return output_file

def subir_resultados_s3(bucket_name, region_name, output_file, s3_path):
    # Usamos boto3 directo para subir cualquier archivo (CSV, JSON, etc)
    s3 = boto3.client('s3', region_name=region_name)
    s3.upload_file(output_file, bucket_name, s3_path)
    print(f"Archivo CSV subido a S3: s3://{bucket_name}/{s3_path}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline scraping/tagging con S3 y OpenAI")
    parser.add_argument('--scrape', action='store_true', help='Ejecutar scraping y subir imágenes a S3')
    parser.add_argument('--tag', action='store_true', help='Ejecutar tagging de imágenes en S3')
    parser.add_argument('--upload', action='store_true', help='Subir el CSV de resultados taggeados a S3')
    parser.add_argument('--bucket', type=str, default=None, help='Nombre del bucket de S3')
    parser.add_argument('--region', type=str, default=None, help='Región de S3')
    parser.add_argument('--prefix', type=str, default="s3-obligatorio-mldata/scraped_data/images/", help='Prefijo de imágenes en S3')
    parser.add_argument('--output', type=str, default="etiquetas_octogonos.csv", help='Archivo local para guardar etiquetas')
    parser.add_argument('--s3_output', type=str, default="scraped_data/results/etiquetas_octogonos.csv", help='Ruta destino en S3 para el CSV')
    args = parser.parse_args()

    # bucket y region: del config si no vienen por argumento
    if not args.bucket or not args.region:
        config = load_settings("Storage")
        args.bucket = args.bucket or config["S3"]["Bucket"]
        args.region = args.region or config["S3"]["Region"]

    # Mostramos config real utilizada
    logger.info(f"Usando bucket: {args.bucket} | Región: {args.region}")

    if args.scrape:
        logger.info("[INFO] Ejecutando scraping...")
        run_scraper()

    if args.tag:
        logger.info("[INFO] Ejecutando tagging...")
        output_file = run_tagger(args.bucket, args.region, args.prefix, args.output)
    else:
        output_file = args.output

    if args.upload:
        logger.info("[INFO] Subiendo resultados a S3...")
        subir_resultados_s3(args.bucket, args.region, output_file, args.s3_output)

if __name__ == "__main__":
    main()
