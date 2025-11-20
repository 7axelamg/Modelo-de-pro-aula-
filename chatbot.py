from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import re
import logging

app = FastAPI()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str = None
    error: str = None
    details: str = None

def remove_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def construir_prompt(mensaje_usuario):
    return f"""
Eres el asistente virtual oficial del Hotel Quantum Gateway. Estás integrado DENTRO del sitio web del hotel. 

⚠️ INSTRUCCIONES ESTRICTAS (OBLIGATORIAS):
- JAMÁS digas: “visita nuestra web”, “entra al sitio”, “búscanos en Google”, ni proporciones URLs.
- El usuario YA está navegando dentro de la página, así que explícale cómo hacer cada cosa usando:
  menú, botones, secciones, formularios o paneles del sitio.
- Responde SIEMPRE en español neutro, tono profesional, cálido y breve (máx. 5–7 líneas).
- No inventes precios, políticas específicas, horarios exactos ni información no confirmada.
- Si el usuario pide datos que requieren sistemas internos (reservas, códigos, pagos, etc.):
  → Explica el procedimiento general y ofrece guiarlo.
- Si el usuario pregunta cómo hacer algo (reservar, cancelar, pagar, contactar):
  → Explica el proceso con pasos concretos dentro de esta misma página.
- No repitas texto, no des listas mayores de 6 pasos y no incluyas URLs.
- Cuando sea oportuno, ofrece ayuda adicional (“Si quieres, puedo guiarte paso a paso.”).

Ejemplo de estilo correcto:
“Para reservar, abre la sección Reservas en el menú superior, elige tus fechas y selecciona una habitación disponible. Luego completa tus datos y confirma.”

Ahora responde de manera útil, clara y profesional al siguiente mensaje del cliente:
{mensaje_usuario}
"""



@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:

        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
        
        env = os.environ.copy()
        env["USERPROFILE"] = os.path.expanduser("~")  
        env["OLLAMA_NO_TTY"] = "1"
        env["OLLAMA_NUM_THREADS"] = "2"

        prompt = construir_prompt(req.message.strip())

        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",  
            env=env,
            timeout=60
        )

        if result.returncode != 0:
            logger.error(f"Error en ollama: {result.stderr}")
            raise HTTPException(status_code=500, detail="Error al ejecutar el modelo")

        raw_response = result.stdout.strip()
        clean_response = remove_ansi(raw_response)


        utf8_response = clean_response


        user_message_lower = req.message.lower()
        
        if not utf8_response or "no tengo acceso" in utf8_response.lower():
            if any(word in user_message_lower for word in ["cancelar", "cancelación", "cancelaste"]):
                return ChatResponse(
                    response=(
                        "Para cancelar tu reserva, comunícate directamente con la recepción del Hotel. "
                        "Puedes hacerlo llamando al número de contacto del hotel o a través del formulario de cancelación "
                        "disponible en nuestro sitio web. Ten a mano tu número de reserva y los datos de tu estadía."
                    )
                )
            return ChatResponse(response="Para esa información específica, por favor contáctanos directamente al WhatsApp del hotel.")

        return ChatResponse(response=utf8_response)

    except HTTPException:
        raise
    except subprocess.TimeoutExpired:
        logger.error("Timeout al ejecutar ollama")
        raise HTTPException(status_code=504, detail="Timeout en la respuesta del modelo")
    except Exception as e:
        logger.error(f"Excepción inesperada: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
