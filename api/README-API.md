# Food Octagon Detection API

Esta aplicación FastAPI detecta octógonos de advertencia en imágenes de empaquetado de alimentos para determinar si los productos alimenticios son saludables o no saludables en base a si tienen o no tienen octógonos.

## Información del modelo

- **Modelo**: LeNet_1 Convolutional Neural Network
- **Input**: 500x500 RGB images
- **Clases**: 
  - `0` = sin_octogono (healthy - no warning octagon)
  - `1` = con_octogono (unhealthy - warning octagon detected)
- **Framework**: PyTorch

## Setup

### Prerequisitos
- Python 3.8+
- PyTorch
- FastAPI

### Instalación

1. **Navegar al directorio de la API:**
   ```bash
   cd api
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual:**
   ```bash
   # En Windows:
   venv\Scripts\activate
   
   # En macOS/Linux:
   source venv/bin/activate
   ```

4. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Asegurar que el archivo del modelo esté en su lugar:**
   ```bash
   # Verificar si el modelo existe
   ls -la model/letnet_model_1.pth
   ```

## Ejecutando la API

### Opción 1: Ejecución Local

#### Iniciar servidor FastAPI:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Iniciar interfaz Gradio:
```bash
python gradio_app.py
```

### Opción 2: Ejecución con Docker (Recomendado)

#### Requisitos Previos:
- Docker instalado en tu sistema
- Docker Compose disponible

#### 🚀 Iniciar la Aplicación Completa:

**Opción A: Ejecución en segundo plano (recomendado)**
```bash
# Construir imágenes y ejecutar servicios en modo detached
docker-compose up --build -d

# Verificar que los servicios estén funcionando
docker-compose ps
```

**Opción B: Ejecución con logs en tiempo real**
```bash
# Ejecutar y ver logs simultáneamente
docker-compose up --build
```

#### 📊 Monitoreo y Gestión:

**Ver logs de los servicios:**
```bash
# Ver logs de ambos servicios
docker-compose logs -f

# Ver logs solo del servicio FastAPI
docker-compose logs -f fastapi

# Ver logs solo del servicio Gradio
docker-compose logs -f gradio
```

**Verificar estado de los contenedores:**
```bash
# Ver estado actual de todos los servicios
docker-compose ps

# Ver información detallada de los contenedores
docker-compose ps -a
```

#### ⚙️ Comandos de Gestión Avanzados:

**Reiniciar servicios:**
```bash
# Reiniciar todos los servicios
docker-compose restart

# Reiniciar solo FastAPI
docker-compose restart fastapi

# Reiniciar solo Gradio
docker-compose restart gradio
```

**Reconstruir imágenes:**
```bash
# Reconstruir todas las imágenes sin cache
docker-compose build --no-cache

# Reconstruir solo FastAPI
docker-compose build --no-cache fastapi

# Reconstruir solo Gradio
docker-compose build --no-cache gradio
```

**Ejecutar servicios individuales:**
```bash
# Ejecutar solo el servicio FastAPI
docker-compose up fastapi

# Ejecutar solo el servicio Gradio
docker-compose up gradio
```

#### 🛑 Detener y Limpiar:

**Detener servicios:**
```bash
# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Detener y eliminar imágenes
docker-compose down --rmi all
```

**Limpieza completa:**
```bash
# Eliminar contenedores, redes, volúmenes e imágenes
docker-compose down --rmi all -v --remove-orphans

# Limpiar recursos Docker no utilizados
docker system prune -f
```

#### 🔧 Solución de Problemas:

**Si los servicios no inician correctamente:**
```bash
# Verificar logs de errores
docker-compose logs

# Reconstruir desde cero
docker-compose down --rmi all -v
docker-compose up --build -d
```

**Si hay problemas de puertos:**
```bash
# Verificar qué puertos están en uso
lsof -i :8000
lsof -i :7860

# Cambiar puertos en docker-compose.yml si es necesario
```

**Verificar conectividad entre servicios:**
```bash
# Probar que FastAPI responde
curl http://localhost:8000/health

# Probar que Gradio responde
curl http://localhost:7860
```

### Disponible en:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc
- **Gradio UI**: http://localhost:7860

## API Endpoints

### 1. Health Check
```http
GET /health
```
Verificar si la API y el modelo están funcionando correctamente.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "LeNet_1 model loaded successfully on cpu"
}
```

### 2. Model Information
```http
GET /model/info
```
Get detailed information about the loaded model.

**Response:**
```json
{
  "model_type": "LeNet_1",
  "input_size": [500, 500],
  "classes": ["sin_octogono", "con_octogono"],
  "device": "cpu",
  "loaded": true
}
```

### 3. Predicción de Imagen Única
```http
POST /predict/single
```
Sube una imagen de comida individual para la detección de octágonos.

**Parameters:**
- `file`: Image file (jpg, png, etc.)

**Response:**
```json
{
  "filename": "food_image.jpg",
  "has_octagon": false,
  "is_healthy": true,
  "confidence": 0.92,
  "message": "✅ No warning octagon found - Healthy food (confidence: 92.00%)"
}
```

### 4. Predicción de Imágenes en Lotes
```http
POST /predict/batch
```
Sube múltiples imágenes de comida para la detección de octágonos en lotes (máximo 10 archivos).

**Parameters:**
- `files`: List of image files

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "has_octagon": true,
      "is_healthy": false,
      "confidence": 0.87,
      "message": "⚠️ Warning octagon detected - Unhealthy food (confidence: 87.00%)"
    },
    {
      "filename": "image2.jpg",
      "has_octagon": false,
      "is_healthy": true,
      "confidence": 0.95,
      "message": "✅ No warning octagon found - Healthy food (confidence: 95.00%)"
    }
  ],
  "total_processed": 2,
  "healthy_count": 1,
  "unhealthy_count": 1
}
```

## Ejemplos de Uso

### Usando curl:

**Single image prediction:**
```bash
curl -X POST "http://localhost:8000/predict/single" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/food_image.jpg"
```

**Health check:**
```bash
curl -X GET "http://localhost:8000/health"
```

### Usando Python requests:

```python
import requests

# Single image prediction
with open('food_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/single', files=files)
    result = response.json()
    print(f"Has octagon: {result['has_octagon']}")
    print(f"Is healthy: {result['is_healthy']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Usando la Documentación Interactiva:

1. Ir a http://localhost:8000/docs
2. Haz clic en cualquier endpoint para expandirlo
3. Haz clic en "Try it out"
4. Sube tu(s) imagen(es)
5. Haz clic en "Execute" para ver los resultados

## Detalles del Modelo

La API utiliza una CNN LeNet_1 entrenada para clasificar imágenes de empaquetado de alimentos:

- **Preprocesamiento de input**: Redimensionar a 500x500, convertir a tensor
- **Output**: Clasificación binaria (octágono/sin octágono)
- **Confianza**: Probabilidad softmax de la clase predicha

## Troubleshooting

### Problemas Comunes:

1. **Modelo no se carga:**
   - Asegúrate de que `letnet_model_1.pth` esté en el directorio `model/`
   - Verifica los permisos del archivo

2. **Errores de importación:**
   - Asegúrate de que todas las dependencias estén instaladas: `pip install -r requirements.txt`

3. **Errores al subir imágenes:**
   - Formatos soportados: JPG, PNG, JPEG
   - Tamaño máximo de archivo: Verifica los límites de tu sistema

4. **Puerto ya en uso:**
   ```bash
   # Usa un puerto diferente
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

## Desarrollo

### Estructura del Proyecto
```
api/
├── main.py              # Punto de entrada de la app FastAPI
├── model/
│   ├── __init__.py
│   ├── predictor.py     # Lógica de inferencia del modelo
│   └── letnet_model_1.pth # Modelo entrenado
├── routes/
│   ├── __init__.py
│   └── prediction.py    # Endpoints de la API
├── schemas.py           # Modelos Pydantic
├── requirements.txt     # Dependencias
├── gradio_app.py        # Interfaz gráfica Gradio
├── README-API.md        # Este archivo
├── documentation/       # Documentación de la API
│   ├── fastapi_openapi_spec.json  # Especificación OpenAPI en JSON
│   ├── fastapi_docs.html          # Documentación Swagger UI
│   └── fastapi_redoc.html         # Documentación ReDoc
├── Dockerfile           # Dockerfile para FastAPI
├── Dockerfile.gradio    # Dockerfile para Gradio
├── docker-compose.yml   # Orquestación de servicios
└── .dockerignore        # Archivos a ignorar en Docker
```

## Interfaz Gráfica con Gradio

La aplicación incluye una interfaz gráfica desarrollada con **Gradio** para facilitar la interacción con la API de detección de octógonos.

### ¿Qué permite la app Gradio?
- Subir una imagen para predicción individual.
- Subir varias imágenes para predicción en lote.
- Visualizar los resultados de cada imagen (octógono/no octógono, confianza, estado saludable).
- Consultar el estado de la API y la información del modelo.

### ¿Cómo usar la app Gradio?
1. Asegúrate de que la API esté corriendo en http://localhost:8000
2. Ejecuta la app Gradio:
   ```bash
   python gradio_app.py
   ```
3. Abre tu navegador en: http://localhost:7860
4. Usa las pestañas para predicción individual o por lote.

### Ejemplo de uso
- Sube una o varias imágenes de empaques de alimentos.
- Haz clic en "Analizar Imagen(es)".
- Observa los resultados en la tabla y el resumen.

### Características
- Interfaz amigable y moderna.
- Resultados claros y visuales.
- Soporte para imágenes JPG y PNG.
- Resúmenes de salud y confianza del modelo.

## Documentación de la API

La API incluye documentación completa descargable en múltiples formatos para facilitar el desarrollo y la integración.

### Archivos de Documentación Disponibles

#### 📄 `documentation/fastapi_openapi_spec.json`
- **Formato**: Especificación OpenAPI 3.0 en JSON
- **Uso**: Importar en herramientas como Postman, Insomnia, o generadores de código
- **Contenido**: Definiciones completas de endpoints, esquemas, ejemplos y respuestas
- **Tamaño**: ~4.6 KB

#### 🌐 `documentation/fastapi_docs.html`
- **Formato**: Documentación Swagger UI interactiva
- **Uso**: Abrir en cualquier navegador web para explorar la API
- **Características**: Interfaz interactiva para probar endpoints directamente
- **Tamaño**: ~958 bytes

#### 📖 `documentation/fastapi_redoc.html`
- **Formato**: Documentación ReDoc alternativa
- **Uso**: Vista alternativa más legible de la documentación
- **Características**: Diseño limpio y fácil de navegar
- **Tamaño**: ~910 bytes

### Cómo Usar la Documentación

#### Para Desarrolladores:
```bash
# Importar especificación OpenAPI en Postman
# 1. Abrir Postman
# 2. File > Import
# 3. Seleccionar documentation/fastapi_openapi_spec.json
# 4. ¡Listo! Todos los endpoints estarán disponibles
```

#### Para Ver la Documentación Web:
```bash
# Abrir documentación Swagger UI
open documentation/fastapi_docs.html

# O abrir documentación ReDoc
open documentation/fastapi_redoc.html
```

#### Para Generar Código Cliente:
```bash
# Usar la especificación OpenAPI con herramientas como:
# - openapi-generator-cli
# - swagger-codegen
# - nswag (para .NET)
```

### Actualizar la Documentación

Para regenerar la documentación después de cambios en la API:

```bash
# 1. Asegúrate de que la API esté corriendo
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 2. Descarga la nueva documentación
curl -s http://localhost:8000/openapi.json > documentation/fastapi_openapi_spec.json
curl -s http://localhost:8000/docs > documentation/fastapi_docs.html
curl -s http://localhost:8000/redoc > documentation/fastapi_redoc.html
```

---

**Nota**: Esta API y la interfaz Gradio están diseñadas para análisis de empaques de alimentos y detección de octógonos de advertencia nutricional según la normativa vigente en Uruguay y otros países.

---

### Agregar nuevos endpoints
1. Define nuevas rutas en `routes/prediction.py`
2. Agrega los esquemas correspondientes en `schemas.py`
3. Actualiza este README con la documentación