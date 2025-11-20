from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional, Dict
import hashlib
import json
from datetime import datetime, timedelta

app = FastAPI()

# Configurar logging optimizado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache mejorado con expiración
class ResponseCache:
    def __init__(self, max_size=200, ttl_minutes=60):
        self.cache: Dict[str, dict] = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['response']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: str):
        if len(self.cache) >= self.max_size:
            # Eliminar el más antiguo
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }

response_cache = ResponseCache()

# Thread pool optimizado
thread_pool = ThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="ollama_worker"
)

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str = None
    error: str = None
    details: str = None
    cached: bool = False

# Compilar regex una sola vez
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Base de conocimiento mejorada de la página
PAGE_KNOWLEDGE = {
    "estructura": {
        "secciones_principales": [
            "Inicio - Presentación de Quantum Gateway",
            "Planes - Suscripciones para hoteles", 
            "Reservas - Sistema de booking",
            "Mi Cuenta - Gestión personal",
            "Soporte - Ayuda y contacto"
        ],
        "menu_superior": ["Inicio", "Planes", "Reservas", "Mi Cuenta", "Soporte"],
        "acciones_principales": ["Suscribirse", "Reservar", "Consultar", "Administrar"]
    },
    "flujos_usuarios": {
        "nuevo_usuario": "Inicio → Planes → Suscribirse → Mi Cuenta → Reservas",
        "usuario_existente": "Mi Cuenta → Reservas → Mis Reservas → Gestión",
        "administrador": "Mi Cuenta → Panel Admin → Gestión Hotel"
    },
    "elementos_ui": {
        "botones": ["Suscribirse", "Reservar", "Buscar", "Confirmar", "Cancelar", "Ver Detalles", "Contactar"],
        "formularios": ["Registro", "Login", "Reserva", "Pago", "Contacto"],
        "paneles": ["Mi Perfil", "Mis Reservas", "Facturación", "Soporte"]
    }
}

# Respuestas inteligentes predefinidas
INTELLIGENT_RESPONSES = {
    "saludo": "¡Hola! Soy tu asistente de Quantum Gateway. Veo que estás en nuestra plataforma. ¿Necesitas ayuda con reservas, planes hoteleros o guía en alguna sección específica?",
    
    "despedida": "¡Fue un gusto ayudarte! Recuerda que puedes navegar por el menú superior para acceder a todas las funciones. ¡Hasta pronto!",
    
    "agradecimiento": "¡De nada! Estoy aquí para ayudarte en lo que necesites. Si tienes más preguntas sobre la plataforma, no dudes en consultarme.",
    
    "planes_generico": "Nuestros planes de administración hotelera incluyen gestión de reservas, panel de control, reportes y soporte. Para ver detalles específicos y precios, ve al menú 'Planes' donde podrás comparar características y suscribirte al que mejor se adapte a tu hotel.",
    
    "reservas_generico": "El sistema de reservas te permite buscar disponibilidad, seleccionar habitaciones y completar tu booking. Desde 'Reservas' en el menú superior puedes ingresar fechas, ver opciones y confirmar. ¿Te guío paso a paso?",
    
    "cancelacion": "Para cancelaciones, accede a 'Mi Cuenta' → 'Mis Reservas', selecciona la reserva y usa la opción 'Cancelar'. Si necesitas ayuda con el proceso, puedo explicarte cada paso en pantalla.",
    
    "contacto": "Para atención personalizada, ve a 'Soporte' en el menú donde encontrarás nuestro WhatsApp, email y formulario de contacto. Estamos para ayudarte.",
    
    "cuenta": "En 'Mi Cuenta' gestionas tu perfil, ves historial de reservas, facturas y configuraciones. Es tu centro de control personal."
}

# Patrones de preguntas frecuentes con respuestas optimizadas
QUESTION_PATTERNS = {
    r"(hola|buenos días|buenas tardes)": "saludo",
    r"(gracias|thanks)": "agradecimiento", 
    r"(adios|chao|hasta luego)": "despedida",
    r"(plan|suscripción|precio|tarifa)": "planes_generico",
    r"(reservar|booking|habitación|alojamiento)": "reservas_generico",
    r"(cancelar|anular)": "cancelacion",
    r"(contacto|soporte|ayuda|whatsapp)": "contacto",
    r"(cuenta|perfil|mis datos)": "cuenta",
    r"(como funciona|qué es|explica)": "explicacion_general"
}

def remove_ansi(text: str) -> str:
    """Elimina códigos ANSI del texto de forma eficiente"""
    return ANSI_ESCAPE.sub('', text)

def generar_hash_mensaje(message: str) -> str:
    """Genera hash único para el mensaje"""
    return hashlib.md5(message.lower().strip().encode()).hexdigest()

def detectar_intencion(message: str) -> Optional[str]:
    """Detecta la intención del mensaje para respuesta rápida"""
    message_lower = message.lower()
    
    for pattern, response_key in QUESTION_PATTERNS.items():
        if re.search(pattern, message_lower, re.IGNORECASE):
            return response_key
    return None

@lru_cache(maxsize=100)
def construir_prompt_inteligente(mensaje_usuario: str) -> str:
    """Construye prompt optimizado con contexto de página"""
    
    contexto_pagina = f"""
INFORMACIÓN ACTUAL DE LA PÁGINA QUANTUM GATEWAY:

ESTRUCTURA Y NAVEGACIÓN:
- Menú superior: {', '.join(PAGE_KNOWLEDGE['estructura']['menu_superior'])}
- Secciones principales: {', '.join(PAGE_KNOWLEDGE['estructura']['secciones_principales'])}
- Flujos comunes: {json.dumps(PAGE_KNOWLEDGE['flujos_usuarios'], indent=2)}
- Elementos de interfaz: Botones ({', '.join(PAGE_KNOWLEDGE['elementos_ui']['botones'])}), Formularios ({', '.join(PAGE_KNOWLEDGE['elementos_ui']['formularios'])}), Paneles ({', '.join(PAGE_KNOWLEDGE['elementos_ui']['paneles'])})

REGLAS CRÍTICAS DE RESPUESTA:
1. RESPUESTA SUPER RÁPIDA - Máximo 4-5 líneas, lenguaje directo
2. USA ELEMENTOS REALES de la página - Nombra botones, menús y secciones exactas
3. GUÍA CONCRETA - Indica ruta específica: "Menú → Sección → Botón"
4. CONTEXTO VISUAL - Asume que el usuario VE la pantalla, describe lo que debe hacer
5. EVITA repeticiones y texto innecesario

EJEMPLOS DE RESPUESTAS ÓPTIMAS:
- "Ve a Planes en el menú, compara los 3 paquetes y pulsa Suscribirse en el que elijas"
- "En Reservas: ingresa fechas, selecciona habitación disponible, completa formulario y Confirma"
- "Mi Cuenta → Mis Reservas → Ver Detalles → Botón Cancelar si necesitas anular"

RESPUESTA A SOLICITUD DEL USUARIO:
"""
    
    return contexto_pagina + f"\nUsuario: {mensaje_usuario}\n\nAsistente:"

def procesar_respuesta_rapida(intencion: str, mensaje_original: str) -> Optional[str]:
    """Genera respuesta rápida basada en intención detectada"""
    if intencion in INTELLIGENT_RESPONSES:
        respuesta_base = INTELLIGENT_RESPONSES[intencion]
        
        # Personalizar based on message context
        if "paso a paso" in mensaje_original.lower():
            return respuesta_base + " Sí, te explico cada paso: "
        elif "urgente" in mensaje_original.lower():
            return respuesta_base + " Para atención inmediata, ve directamente a Soporte → Contacto."
        
        return respuesta_base
    
    return None

def run_ollama_optimizado(prompt: str) -> str:
    """Ejecuta Ollama con configuración optimizada para velocidad"""
    try:
        env = os.environ.copy()
        env.update({
            "USERPROFILE": os.path.expanduser("~"),
            "OLLAMA_NO_TTY": "1",
            "OLLAMA_NUM_THREADS": "6",  # Más threads para mayor velocidad
            "OLLAMA_KEEP_ALIVE": "10m",  # Mantener modelo listo
            "OLLAMA_MAX_LOADED_MODELS": "2"
        })
        
        # Modelo optimizado para velocidad
        model_name = "gemma3:12b"  # o "llama3.1:8b" para máxima velocidad
        
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=30,  # Timeout reducido
            cwd=os.path.expanduser("~")  # Directorio de trabajo óptimo
        )

        if result.returncode != 0:
            logger.error(f"Error Ollama: {result.stderr[:200]}")
            return None

        raw_response = result.stdout.strip()
        return remove_ansi(raw_response)

    except subprocess.TimeoutExpired:
        logger.warning("Timeout Ollama - usando respuesta de respaldo")
        return "Estoy optimizando la respuesta. Por favor, revisa en el menú la sección correspondiente o intenta de nuevo."
    except Exception as e:
        logger.error(f"Error Ollama: {str(e)}")
        return None

def mejorar_respuesta_contexto(respuesta: str, mensaje_original: str) -> str:
    """Mejora la respuesta con contexto específico de la página"""
    
    # Asegurar que menciona elementos de la UI
    elementos_ui = PAGE_KNOWLEDGE['elementos_ui']
    
    # Buscar y asegurar referencias a la interfaz
    if "ve a" in respuesta.lower() and "menú" not in respuesta.lower():
        respuesta = respuesta.replace("Ve a ", "Ve al menú ")
    
    # Agregar oferta de ayuda si no está presente
    if "paso a paso" not in respuesta.lower() and "guiar" not in respuesta.lower():
        respuesta += " ¿Necesitas que te guíe paso a paso?"
    
    return respuesta

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        start_time = datetime.now()
        
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío")
        
        message = req.message.strip()
        message_hash = generar_hash_mensaje(message)
        
        logger.info(f"Procesando mensaje: {message[:60]}...")
        
        # PASO 1: Verificar cache (más rápido)
        cached_response = response_cache.get(message_hash)
        if cached_response:
            logger.info(f"Respuesta desde cache - Tiempo: {(datetime.now() - start_time).total_seconds():.2f}s")
            return ChatResponse(response=cached_response, cached=True)
        
        # PASO 2: Detección rápida de intención
        intencion = detectar_intencion(message)
        if intencion:
            respuesta_rapida = procesar_respuesta_rapida(intencion, message)
            if respuesta_rapida:
                # Cachear respuesta rápida
                response_cache.set(message_hash, respuesta_rapida)
                logger.info(f"Respuesta rápida - Intención: {intencion} - Tiempo: {(datetime.now() - start_time).total_seconds():.2f}s")
                return ChatResponse(response=respuesta_rapida)
        
        # PASO 3: Procesamiento con modelo (optimizado)
        prompt = construir_prompt_inteligente(message)
        
        loop = asyncio.get_event_loop()
        raw_response = await loop.run_in_executor(
            thread_pool, 
            run_ollama_optimizado, 
            prompt
        )
        
        if raw_response is None:
            # Respuesta de respaldo inteligente
            fallback_response = "En este momento puedo sugerirte: revisa en el menú superior las secciones disponibles. Para ayuda inmediata, ve a Soporte → Contacto."
            response_cache.set(message_hash, fallback_response)
            return ChatResponse(response=fallback_response)
        
        # PASO 4: Mejorar respuesta con contexto
        final_response = mejorar_respuesta_contexto(raw_response, message)
        
        # Cachear respuesta final
        response_cache.set(message_hash, final_response)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Respuesta generada - Tiempo: {total_time:.2f}s - Longitud: {len(final_response)}")
        
        return ChatResponse(response=final_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Error procesando tu solicitud")

# Endpoints adicionales para inteligencia mejorada
@app.get("/page-knowledge")
async def get_page_knowledge():
    """Endpoint para conocer la estructura de la página"""
    return {
        "knowledge_base": PAGE_KNOWLEDGE,
        "intelligent_responses": list(INTELLIGENT_RESPONSES.keys()),
        "cache_stats": {
            "size": len(response_cache.cache),
            "max_size": response_cache.max_size
        }
    }

@app.get("/cache-stats")
async def get_cache_stats():
    """Estadísticas del cache"""
    return {
        "cache_size": len(response_cache.cache),
        "max_size": response_cache.max_size,
        "ttl_minutes": response_cache.ttl.total_seconds() / 60
    }

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Servicio AI Optimizado iniciado - Respuestas inteligentes activas")
    logger.info(f"📚 Base de conocimiento: {len(PAGE_KNOWLEDGE['estructura']['secciones_principales'])} secciones mapeadas")

@app.on_event("shutdown")
async def shutdown_event():
    thread_pool.shutdown(wait=False)
    logger.info("Servicio detenido - Cache preservado")