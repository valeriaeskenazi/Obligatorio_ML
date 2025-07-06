import gradio as gr
import requests
import json
from PIL import Image
import io
import base64
import os

# Configuración de la API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def call_single_prediction_api(image):
    """Llama al endpoint de predicción individual"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        files = {'file': ('imagen.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/predict/single", files=files)
        if response.status_code == 200:
            result = response.json()
            return {
                "Estado": "✅ Éxito",
                "¿Tiene octógono?": "⚠️ Sí" if result['has_octagon'] else "✅ No",
                "Confianza": f"{result['confidence']:.2%}",
                "Estado de salud": "🚨 No saludable" if result['has_octagon'] else "✅ Saludable",
                "Mensaje": result['message'].replace('Warning octagon detected', '⚠️ Octógono de advertencia detectado').replace('No warning octagon found', '✅ Sin octógono de advertencia')
            }
        else:
            return {
                "Estado": "❌ Error",
                "¿Tiene octógono?": "N/A",
                "Confianza": "N/A",
                "Estado de salud": "N/A",
                "Mensaje": f"Error de la API: {response.status_code}"
            }
    except Exception as e:
        return {
            "Estado": "❌ Error",
            "¿Tiene octógono?": "N/A",
            "Confianza": "N/A",
            "Estado de salud": "N/A",
            "Mensaje": f"Error: {str(e)}"
        }

def call_batch_prediction_api(images):
    """Llama al endpoint de predicción por lote"""
    try:
        if not images:
            return "❌ No se proporcionaron imágenes", []
        files = []
        for i, image in enumerate(images):
            if image is not None:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                files.append(('files', (f'imagen_{i}.jpg', img_byte_arr, 'image/jpeg')))
        if not files:
            return "❌ No se proporcionaron imágenes válidas", []
        response = requests.post(f"{API_BASE_URL}/predict/batch", files=files)
        if response.status_code == 200:
            result = response.json()
            tabla = []
            for item in result['results']:
                tabla.append([
                    item['filename'],
                    "⚠️ Sí" if item['has_octagon'] else "✅ No",
                    f"{item['confidence']:.2%}",
                    "🚨 No saludable" if item['has_octagon'] else "✅ Saludable"
                ])
            resumen = f"""
            📊 **Resultados de la predicción por lote**
            
            **Total procesadas**: {result['total_processed']} imágenes
            **Saludables**: {result['no_octagon_count']} ✅
            **No saludables**: {result['octagon_count']} 🚨
            """
            return resumen, tabla
        else:
            return f"❌ Error de la API: {response.status_code}", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []

def check_api_health():
    """Verifica si la API está corriendo"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return f"✅ Estado de la API: {health_data['status']}\n📝 {health_data['message']}"
        else:
            return f"❌ Error de la API: {response.status_code}"
    except Exception as e:
        return f"❌ No se puede conectar a la API: {str(e)}"

with gr.Blocks(title="Sistema de Detección de Octógonos en Alimentos", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍎 Sistema de Detección de Octógonos en Alimentos")
    gr.Markdown("Subí imágenes de empaques de alimentos para detectar octógonos de advertencia y saber si el producto es saludable o no.")
    
    with gr.Row():
        health_btn = gr.Button("🔍 Verificar estado de la API", variant="secondary")
        health_output = gr.Textbox(label="Estado de la API", interactive=False)
    health_btn.click(check_api_health, outputs=health_output)
    
    with gr.Tab("📸 Predicción individual"):
        with gr.Row():
            with gr.Column():
                single_image_input = gr.Image(label="Subir imagen de alimento", type="pil")
                single_predict_btn = gr.Button("🔍 Analizar imagen", variant="primary")
            with gr.Column():
                single_results = gr.JSON(label="Resultados de la predicción")
        single_predict_btn.click(
            call_single_prediction_api,
            inputs=single_image_input,
            outputs=single_results
        )
    
    with gr.Tab("📚 Predicción por lote"):
        with gr.Row():
            with gr.Column():
                batch_images_input = gr.File(
                    label="Subir varias imágenes",
                    file_count="multiple",
                    file_types=["image"]
                )
                batch_predict_btn = gr.Button("🔍 Analizar imágenes", variant="primary")
            with gr.Column():
                batch_summary = gr.Markdown(label="Resumen")
                batch_results_table = gr.Dataframe(
                    headers=["Imagen", "¿Tiene octógono?", "Confianza", "Estado de salud"],
                    label="Tabla de resultados"
                )
        def process_batch_files(files):
            images = []
            for file in files:
                try:
                    with open(file.name, 'rb') as f:
                        img_data = f.read()
                    image = Image.open(io.BytesIO(img_data))
                    images.append(image)
                except Exception as e:
                    print(f"Error cargando la imagen {file.name}: {e}")
            return call_batch_prediction_api(images)
        batch_predict_btn.click(
            process_batch_files,
            inputs=batch_images_input,
            outputs=[batch_summary, batch_results_table]
        )
    
    with gr.Tab("ℹ️ Información"):
        gr.Markdown("""
        ## Sobre este sistema
        
        Esta aplicación utiliza una red neuronal convolucional **LeNet-1** para detectar octógonos de advertencia en empaques de alimentos.
        
        ### ¿Qué son los octógonos de advertencia?
        Son etiquetas obligatorias en Uruguay y otros países que indican altos niveles de:
        - 🍬 Azúcar
        - 🧈 Grasas
        - 🧂 Sodio
        
        ### ¿Cómo usar?
        1. Predicción individual: subí una imagen para analizar.
        2. Predicción por lote: subí varias imágenes para análisis masivo.
        3. Resultados: obtené el estado de salud y la confianza del modelo.
        
        ### Información del modelo:
        - **Input**: imágenes RGB de 500x500
        - **Output**: clasificación binaria (octógono/sin octógono)
        - **Confianza**: probabilidad de la clase predicha
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
