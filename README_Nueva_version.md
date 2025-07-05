# Paso a paso el proceso de scrappeado y taggeado

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
Enlace al modelo: https://drive.google.com/file/d/1KAo8Ox3J8JzSeOw0KgFyrhJTBtMocGY7/view?usp=sharing

