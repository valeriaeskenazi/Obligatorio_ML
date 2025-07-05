import openai
import base64
import os
from io import BytesIO
from PIL import Image

openai.api_key = os.getenv("OPENAI_API_KEY")

def clasificar_octogono(image_bytes: BytesIO) -> str:
    image = Image.open(image_bytes).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Dado un envase de producto, responde sólo 'con_octogono' o 'sin_octogono' según si hay octógonos de advertencia nutricional visibles. No expliques nada."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content.strip().lower()
