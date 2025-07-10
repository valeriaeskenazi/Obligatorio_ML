# Paso a paso el proceso de scrappeado y taggeado

## Estructura para el armado del Dataset

src/
├── connectors/
│   ├── openai_client.py         # Conexión con la API de OpenAI
│   └── s3_client.py             # Cliente para interactuar con AWS S3
│
├── scrapers/
│   └── disco.py                 # Lógica para scrapear productos de Disco
│
├── scripts/
│   └── empty_s3_bucket.py       # Script auxiliar para vaciar un bucket de S3
│
├── settings/
│   ├── config.yml               # Configuraciones generales del proyecto
│   ├── logger.py                # Configuración de logging
│   └── settings.py              # Lectura de configuraciones
│
├── structs/
│   ├── product.py               # Definición de estructura de productos
│   └── storage_type.py          # Tipos de almacenamiento disponibles
│
├── tags/
│   └── etiquetador.py           # Lógica para el etiquetado automático
│
├── utils/
│   └── io_utils.py              # Utilidades para lectura y escritura de archivos
│
├── main.py                      # Script principal (no utilizado en la ejecución estándar)
└── pipeline.py                  # Script principal de ejecución: scraping, tagging y subida a S3




## Instalacion de dependencias

pip install -r requirements.txt

python -m playwright install


## Conexion con AWS academy
Con aws configure me conecto con AWS

Creo un archivo de notas para poder exportar rapidamente las credenciales:

export AWS_ACCESS_KEY_ID="TU_AWS_ACCESS_KEY"

export AWS_SECRET_ACCESS_KEY="TU_AWS_SECRET"

export AWS_SESSION_TOKEN="TU_AWS_SESSION_TOKEN"

export AWS_DEFAULT_REGION="us-east-1"

## Conexion con OPENAI

En el archivo notas, pego mi clave para tenerla a mano:

export OPENAI_API_KEY="sk-xxxxx"

## Creacion de bucket

Creo un nuevo buket en s3 con el nombre: '1000-imagenes-scrapper-obligatorio-ml'

Cambiamos en config.yml a este nombre

## ejecucion

corro:

python src/pipeline.py --scrape --tag --upload --bucket 1000-imagenes-scrapper-obligatorio-ml   


## Problemas

Hubo un erro en el prefijo de guardado, dado que quedo scrappeado utilizamos esa info del buket para taggear:

python src/pipeline.py --tag --upload --bucket 1000-imagenes-scrapper-obligatorio-ml --prefix data/scraped_data/images/

## Conexion a repo

git init

git remote add origin https://github.com/valeriaeskenazi/Obligatorio_ML.git

## Subir cambios

Antes de subir los cambios, eliminamos el archivo "notas.txt" con claves.

git add .

git commit -m "Mi primer commit"

git push -u origin main

## Problemas con push a github

Al haber trabajado desde una nueva carpeta comenzo a haber problemas e versiones, para solucionarlo:

git pull origin main

git pull origin main --no-rebase

git pull origin main --allow-unrelated-histories

git pull origin main --no-rebase --allow-unrelated-histories

Ahora si, el problema estaba en el README, se cambia nombre y se proceede de manera habitual


# Clasificador
## Modelo 1
Enlace al modelo 2: https://drive.google.com/file/d/1KAo8Ox3J8JzSeOw0KgFyrhJTBtMocGY7/view?usp=sharing

Enlace al modelo 2: https://drive.google.com/file/d/13m-CeDUYAfDH0dMW73GdpAol9G_0CmEX/view?usp=drive_link

(este modelo por su peso, se encuentra en github, dentro de la carpeta de API/model)

``` bash
git checkout --orphan fresh-main
git add -A
git commit -m "Versión completamente limpia, sin historial ni replace.txt"
git branch -D main
git branch -m main
git remote remove origin
git remote add origin https://github.com/valeriaeskenazi/Obligatorio_ML.git
git push --force origin main
```

```bash
git add Clasificador/Clasificador_Imagenes.ipynb
git commit -m "Notebook limpia sin credenciales"
```

# API

Luego de haber realizado las instalaciones segun el README.API

## Subida de Modelo 1 a S3

Conexion con AWS como de costumbre

Conexion con el bucket:

```bash
aws s3 mb s3://s3-682317053885-modelos-obligatorio-ml --region us-east-1
```

Subir archivo:

```bash
aws s3 cp letnet_model_1.pth s3://s3-682317053885-modelos-obligatorio-ml/letnet_model_1.pth
```

## Crear ECR 

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 682317053885.dkr.ecr.us-east-1.amazonaws.com
```

### Cosntruir la imagen

```bash
docker-compose up --build -d
```

### Etiquetar la imagen

```bash
docker tag api-gradio:latest 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-gradio:latest

docker tag api-fastapi:latest 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-fastapi:latest
```

### Subir la imagen a ECR

```bash
docker push 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-gradio

docker push 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-fastapi
```

## Conectarse a EC2

```bash
chmod 400 Obligatorio-ML.pem
ssh -i Obligatorio-ML.pem ec2-user@13.219.77.245
```

### correr con docker
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 682317053885.dkr.ecr.us-east-1.amazonaws.com

docker pull 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-fastapi:latest
docker pull 682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-gradio:latest

docker network create octagon-network
```

### Correr fastapi

```bash
docker run -d --name api-fastapi \
  --network octagon-network \
  -p 8000:8000 \
  -v ~/model:/app/model \
  682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-fastapi:latest
```

### Correr gradio
``` bash
docker run -d --name api-gradio \
  --network octagon-network \
  -p 7860:7860 \
  682317053885.dkr.ecr.us-east-1.amazonaws.com/ecr-api-gradio:latest
```

## En EC2 
Se agregra en grupos de seguridad, reglas de entrada los puertos 7860 y 8000


## Entrar a la api
http://13.219.77.245:7860
http://13.219.77.245:8000/docs

