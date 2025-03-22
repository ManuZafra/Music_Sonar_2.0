import os
import hmac
import hashlib
import base64
import time
import requests
import gradio as gr
import soundfile as sf
import librosa
import numpy as np
from smolagents import tool, CodeAgent
from huggingface_hub import InferenceClient
import tempfile

# Cargar claves de entorno
ACR_ACCESS_KEY = os.environ.get("ACR_ACCESS_KEY")
ACR_SECRET_KEY = os.environ.get("ACR_SECRET_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not all([ACR_ACCESS_KEY, ACR_SECRET_KEY, HF_TOKEN]):
    raise ValueError("Faltan variables de entorno necesarias")

# Configuración del modelo LLM
llm = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKEN)

# Wrapper para LLM
class LLMWrapper:
    def __init__(self, client):
        self.client = client

    def __call__(self, prompt, **kwargs):
        print(f"Received prompt: {prompt}")
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict):
                    text = msg.get("text") or msg.get("content")
                    if text:
                        print(f"Extracted text: {text}")
                        return self.client.text_generation(text, max_new_tokens=500, temperature=0.7)
            text = str(prompt[0]) if prompt else str(prompt)
            print(f"Fallback text from list: {text}")
            return self.client.text_generation(text, max_new_tokens=500, temperature=0.7)
        text = str(prompt)
        print(f"Using string directly: {text}")
        return self.client.text_generation(text, max_new_tokens=500, temperature=0.7)

# Herramienta de reconocimiento
@tool
def recognize_song(audio_path: str) -> dict:
    """
    Recognize a song from an audio file using the ACRCloud API.

    Args:
        audio_path (str): Path to the audio file to be recognized (e.g., 'temp_audio.wav').

    Returns:
        dict: Dictionary containing song details if successful, or an error message if failed.
              Possible keys on success: 'title', 'artist', 'album', 'release_date'.
              On failure: {'error': str}.
    """
    try:
        audio_data, sample_rate = sf.read(audio_path)
        if sample_rate != 8000:
            print(f"Warning: Sample rate is {sample_rate}, expected 8000 Hz.")

        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        timestamp = str(int(time.time()))
        data_type = "audio"
        signature_version = "1"
        string_to_sign = f"POST\n/v1/identify\n{ACR_ACCESS_KEY}\n{data_type}\n{signature_version}\n{timestamp}"
        sign = base64.b64encode(hmac.new(ACR_SECRET_KEY.encode("ascii"), string_to_sign.encode("ascii"), digestmod=hashlib.sha1).digest()).decode("ascii")

        files = {"sample": (audio_path, open(audio_path, "rb"), "audio/wav")}
        data = {
            "access_key": ACR_ACCESS_KEY,
            "data_type": data_type,
            "signature_version": signature_version,
            "signature": sign,
            "sample_bytes": os.path.getsize(audio_path),
            "timestamp": timestamp
        }

        response = requests.post(url, files=files, data=data)
        response_data = response.json()

        if response_data.get("status", {}).get("code") == 0:
            if "metadata" in response_data and "music" in response_data["metadata"] and response_data["metadata"]["music"]:
                metadata = response_data["metadata"]["music"][0]
                return {
                    "title": metadata.get("title", "Unknown"),
                    "artist": metadata["artists"][0]["name"] if metadata.get("artists") else "Unknown",
                    "album": metadata.get("album", {}).get("name", "Unknown"),
                    "release_date": metadata.get("release_date", "Unknown")
                }
            else:
                return {"error": "No se encontraron coincidencias para el audio proporcionado"}
        else:
            return {"error": response_data.get("status", {}).get("msg", "Error desconocido en ACRCloud")}
    except Exception as e:
        print(f"Error in recognize_song: {str(e)}")
        return {"error": str(e)}

# Configuración del agente
agent = CodeAgent(tools=[recognize_song], model=LLMWrapper(llm), additional_authorized_imports=["boto3"])

# Información dinámica del artista con LLM
def get_artist_info(artist_name: str) -> str:
    prompt = f"Dame una breve biografía de {artist_name}, destacando su carrera y estilo musical, en español."
    try:
        return llm.text_generation(prompt, max_new_tokens=200, temperature=0.7)
    except Exception as e:
        return f"No se pudo obtener info de {artist_name}: {str(e)}"

# Curiosidades dinámicas con LLM
def get_curiosities(artist_name: str) -> str:
    prompt = f"En español, lista 2-3 datos interesantes sobre {artist_name} relacionados con su música o carrera en formato:\n1. [Dato 1]\n2. [Dato 2]\n3. [Dato 3]"
    try:
        return llm.text_generation(prompt, max_new_tokens=200, temperature=0.7)
    except Exception as e:
        return f"No hay curiosidades disponibles para {artist_name}: {str(e)}"

# Procesar audio
def process_audio(audio):
    if audio is None:
        return "No se recibió audio.", None

    target_sr = 8000
    audio_data = audio[1].astype(np.float32) / 32768.0
    audio_data = librosa.resample(audio_data, orig_sr=audio[0], target_sr=target_sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_path, audio_data, target_sr)
        result = recognize_song(temp_path)
    os.remove(temp_path)

    if "error" in result:
        query = f"No se pudo identificar la canción: {result['error']}. ¿Qué puedo hacer?"
        try:
            agent_response = agent.run(query)
            return agent_response, None
        except Exception as e:
            return f"Error al consultar al agente: {str(e)}", None

    song_title = result["title"]
    artist_name = result["artist"]
    album = result["album"]
    release_date = result["release_date"]
    artist_info = get_artist_info(artist_name)
    curiosities = get_curiosities(artist_name)

    output = (
        "<style>.large-text { font-size: 24px; line-height: 1.5; }</style>\n"
        "<h2>🎵 Detalles de la Canción 🎵</h2>\n"
        "<div class='large-text'>\n"
        "──────────────────────\n"
        f"🎵 Título: {song_title}\n\n"
        f"🎤 Artista: {artist_name}\n\n"
        f"📅 Lanzamiento: {release_date}\n\n"
        f"📀 Álbum: {album}\n\n"
        f"🏷️ Sello: No disponible en ACRCloud\n\n"
        f"🎧 Género: No disponible en ACRCloud\n"
        "──────────────────────\n\n"
        f"👤 **Sobre {artist_name}** 👤\n"
        "──────────────────────\n"
        f"{artist_info}\n\n"
        "──────────────────────\n"
        "✨ **Curiosidades y Anécdotas** ✨\n"
        "──────────────────────\n"
        f"{curiosities}\n"
        "</div>"
    )
    return output, artist_name

# Chat con el agente
def chat_with_llm(message, history, artist_name):
    if not artist_name:
        return "Primero identifica una canción para chatear sobre el artista."
    query = f"Pregunta sobre {artist_name}: {message}"
    try:
        response = agent.run(query)
        return f"**Tú**: {message}\n**Respuesta**: {response}"
    except Exception as e:
        return f"**Tú**: {message}\n**Error**: {str(e)}"

# Interfaz de Gradio
with gr.Blocks(title="Music Sonar 2.0") as interface:
    gr.Markdown("# 🎧 Music Sonar 2.0")
    gr.Markdown("Sube o graba un audio para descubrir la canción y más sobre el artista.")

    audio_input = gr.Audio(type="numpy", label="Graba o sube un audio", recording=True)
    submit_btn = gr.Button("Identificar")
    output_text = gr.Markdown(label="Resultados")
    artist_state = gr.State()

    with gr.Column():
        gr.Markdown("### 💬 Chatea sobre el artista")
        chat_output = gr.Textbox(label="Conversación", lines=10, interactive=False)
        chat_input = gr.Textbox(label="Pregunta algo", placeholder="E.g., ¿Qué inspira a este artista?")
        chat_submit = gr.Button("Enviar")

    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=[output_text, artist_state])
    chat_submit.click(fn=chat_with_llm, inputs=[chat_input, chat_output, artist_state], outputs=chat_output)

if __name__ == "__main__":
    interface.launch(share=True)
