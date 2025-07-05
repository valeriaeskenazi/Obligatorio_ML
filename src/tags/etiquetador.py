import os
from connectors.s3_client import S3Client
from connectors.openai_client import clasificar_octogono
from settings.settings import load_settings
from settings.logger import custom_logger
from utils.io_utils import guardar_csv
import boto3 

logger = custom_logger("Etiquetador")

def main():
    # Cargar configuración desde config.yml
    config = load_settings("Storage")
    bucket_name = config["S3"]["Bucket"]
    region_name = config["S3"]["Region"]

    # Inicializar cliente S3
    s3 = S3Client(bucket_name=bucket_name, region_name=region_name)

    claves = s3.list_files(prefix="s3-obligatorio-mldata/scraped_data/images/")
    resultados = []

    for key in claves:
        try:
            logger.info(f"Procesando imagen: {key}")
            imagen_bytes = s3.download_image(key)
            etiqueta = clasificar_octogono(imagen_bytes)
            resultados.append((key, etiqueta))
        except Exception as e:
            logger.error(f"❌ Error procesando {key}: {e}")

    # Guardar resultados
    output_file = "etiquetas_octogonos.csv"
    guardar_csv(resultados, output_file)
    logger.info(f"\n✅ Proceso terminado. Resultados guardados en {output_file}")

if __name__ == "__main__":
    main()
