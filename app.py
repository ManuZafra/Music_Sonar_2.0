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

# Clase auxiliar para simular un objeto con .content
class SimpleResponse:
    def __init__(self, content):
        self.content = content

# Cargar claves de entorno
ACR_ACCESS_KEY = os.environ.get("ACR_ACCESS_KEY")
ACR_SECRET_KEY = os.environ.get("ACR_SECRET_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Verificar claves
if not ACR_ACCESS_KEY or not ACR_SECRET_KEY:
    raise ValueError("Faltan ACR_ACCESS_KEY o ACR_SECRET_KEY en las variables de entorno")
if not HF_TOKEN:
    raise ValueError("Falta HF_TOKEN en las variables de entorno")

# Configuración del modelo LLM usando Hugging Face Inference API
llm = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN
)

# Wrapper para adaptar InferenceClient a smolagents
class LLMWrapper:
    def __init__(self, client):
        self.client = client

    def __call__(self, prompt, **kwargs):
        if isinstance(prompt, list):
            for message in prompt:
                if message.get("role") == "user":
                    user_content = message.get("content")
                    if isinstance(user_content, list):
                        for item in user_content:
                            if item.get("type") == "text":
                                prompt = item.get("text")
                                break
                    else:
                        prompt = user_content
                    break
            else:
                prompt = str(prompt)
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        print(f"Prompt enviado a text_generation: {prompt}")
        response = self.client.text_generation(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7
        )
        print(f"Raw response from text_generation: {response}")

        # Si la respuesta no incluye un bloque de código con final_answer, lo añadimos
        if "```py" not in response and "final_answer" not in response:
            formatted_response = (
                "Thought: La tarea es proporcionar sugerencias sobre qué hacer con la información de la canción identificada. No se requiere ejecución de código adicional, solo devolver la respuesta.\n"
                "Code:\n"
                "```py\n"
                f"final_answer({repr(response)})\n"
                "```<end_code>"
            )
            return SimpleResponse(formatted_response)
        return SimpleResponse(response)

# Definir la herramienta antes del agente
@tool
def recognize_song(audio_path: str) -> dict:
    """
    Recognize a song from an audio file using the ACRCloud API.

    Args:
        audio_path (str): Path to the audio file to be recognized.

    Returns:
        dict: Dictionary containing song title and artist if successful, or an error message if failed.
              - On success: {'title': str, 'artist': str}
              - On failure: {'error': str}

    Example:
        >>> recognize_song("temp_audio.wav")
        {'title': 'No Revolution (Original Mix)', 'artist': 'Joris Voorn'}
    """
    print(f"Recognizing song from: {audio_path}")
    try:
        audio_data, sample_rate = sf.read(audio_path)
        print(f"Sample rate: {sample_rate}, Audio data shape: {audio_data.shape}")
        if sample_rate != 8000:
            print(f"Warning: Sample rate is {sample_rate}, expected 8000 Hz.")

        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        timestamp = str(int(time.time()))
        data_type = "audio"
        signature_version = "1"
        string_to_sign = f"POST\n/v1/identify\n{ACR_ACCESS_KEY}\n{data_type}\n{signature_version}\n{timestamp}"
        sign = base64.b64encode(
            hmac.new(
                ACR_SECRET_KEY.encode("ascii"),
                string_to_sign.encode("ascii"),
                digestmod=hashlib.sha1
            ).digest()
        ).decode("ascii")

        files = {"sample": (audio_path, open(audio_path, "rb"), "audio/wav")}
        data = {
            "access_key": ACR_ACCESS_KEY,
            "data_type": data_type,
            "signature_version": signature_version,
            "signature": sign,
            "sample_bytes": os.path.getsize(audio_path),
            "timestamp": timestamp
        }

        print("Sending request to ACRCloud...")
        response = requests.post(url, files=files, data=data)
        response_data = response.json()
        print(f"ACRCloud response: {response_data}")

        if response_data.get("status", {}).get("code") == 0:
            if "metadata" in response_data and "music" in response_data["metadata"] and response_data["metadata"]["music"]:
                metadata = response_data["metadata"]["music"][0]
                return {
                    "title": metadata["title"],
                    "artist": metadata["artists"][0]["name"]
                }
            else:
                return {"error": "No se encontraron coincidencias para la canción"}
        else:
            return {"error": response_data.get("status", {}).get("msg", "Unknown error")}
    except Exception as e:
        print(f"Error in recognize_song: {str(e)}")
        return {"error": str(e)}

# Configuración del agente con imports autorizados
agent = CodeAgent(
    tools=[recognize_song],
    model=LLMWrapper(llm),
    additional_authorized_imports=["boto3"]
)

def process_audio(audio):
    print("Processing audio...")
    if audio is None:
        print("No audio received")
        return "No se recibió audio."

    target_sr = 8000
    audio_data = audio[1].astype(np.float32) / 32768.0
    audio_data = librosa.resample(audio_data, orig_sr=audio[0], target_sr=target_sr)
    temp_path = "temp_audio.wav"
    print(f"Saving audio to {temp_path}, sample rate: {target_sr}, data shape: {audio_data.shape}")
    sf.write(temp_path, audio_data, target_sr)

    result = recognize_song(temp_path)
    print(f"Recognition result: {result}")

    if "error" in result:
        query = f"No se pudo identificar la canción: {result['error']}. ¿Qué puedo hacer?"
    else:
        query = (
            f"Acabo de identificar la canción '{result['title']}' de {result['artist']} usando ACRCloud. "
            "Dame 5 sugerencias específicas sobre qué puedo hacer con esta información, "
            "como escuchar más música similar, buscar datos del artista, o usarla creativamente."
        )

    print(f"Querying agent with: {query}")
    try:
        agent_response = agent.run(query)
        # Verifica si agent_response tiene .content, si no, usa el texto crudo
        response_text = getattr(agent_response, "content", str(agent_response))
        print(f"Agent response: {response_text}")
        return response_text
    except Exception as e:
        print(f"Error in agent.run: {str(e)}")
        return f"Error al consultar al agente: {str(e)}"

# Interfaz de Gradio
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="numpy", recording=True),
    outputs="text",
    title="Music Sonar 2.0",
    description="Graba audio y descubre qué canción es con ACRCloud y asistencia de IA."
)

if __name__ == "__main__":
    interface.launch(share=True)
