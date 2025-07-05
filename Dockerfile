# Usa Python 3.11 en su versión slim para mantener la imagen liviana
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los requerimientos antes para aprovechar el cache de docker
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto (incluyendo src/scripts y otros necesarios)
COPY . .

# ENTRYPOINT queda como python, así elegís qué script correr al ejecutar el contenedor
ENTRYPOINT ["python"]

# CMD por defecto ejecuta el scrapper, pero al hacer docker run podes sobreescribirlo
CMD ["src/scripts/scrapper.py"]
