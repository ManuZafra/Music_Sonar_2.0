from smolagents import tool
import gradio as gr
import logging
import tempfile
import soundfile as sf
import numpy as np
import os
import requests
import hmac
import hashlib
import base64
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("music_recognition_app")

@tool
def recognize_song(audio_path: str) -> dict:
    """
    Recognize a song from an audio file using the ACRCloud API.
    Args:
        audio_path (str): Path to the audio file to be recognized.
    Returns:
        dict: Dictionary containing song title and artist if successful, or an error message if failed.
    """
    logger.info(f"Recognizing song from: {audio_path}")
    ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
    ACR_SECRET_KEY = os.getenv("ACR_SECRET_KEY")
    if not ACR_ACCESS_KEY or not ACR_SECRET_KEY:
        logger.error("ACRCloud credentials missing")
        return {"error": "ACRCloud credentials not set"}
    if not os.path.exists(audio_path):
        logger.error("Audio file does not exist")
        return {"error": "Audio file does not exist"}

    try:
        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        timestamp = str(int(time.time()))
        string_to_sign = f"POST\n/v1/identify\n{ACR_ACCESS_KEY}\naudio\n1\n{timestamp}"
        sign = hmac.new(ACR_SECRET_KEY.encode(), string_to_sign.encode(), hashlib.sha1).digest()
        signature = base64.b64encode(sign).decode()

        data = {
            "access_key": ACR_ACCESS_KEY,
            "sample_rate": "44100",
            "audio_format": "mp3",
            "signature_version": "1",
            "signature": signature,
            "timestamp": timestamp,
            "data_type": "audio",
            "sample_bytes": str(os.path.getsize(audio_path))
        }
        logger.info(f"String to sign: {string_to_sign}")
        logger.info(f"Request data: {data}")

        with open(audio_path, "rb") as file:
            response = requests.post(url, data=data, files={"sample": file})
        logger.info(f"ACRCloud response: status={response.status_code}, content={response.text}")

        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        result = response.json()
        if result.get("status", {}).get("code", -1) == 0:
            song_info = result["metadata"]["music"][0]
            return {
                "Song": song_info.get("title", "Unknown"),
                "Artist": song_info.get("artists", [{}])[0].get("name", "Unknown")
            }
        return {"error": result["status"]["msg"]}
    except Exception as e:
        logger.error(f"Error in recognize_song: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def process_audio(audio):
    """Procesa el audio grabado y lo envía a la herramienta de reconocimiento."""
    if audio is None:
        logger.info("No audio received")
        return "Please record some audio first"
    try:
        sample_rate, audio_data = audio
        logger.info(f"Audio data: sample_rate={sample_rate}, shape={audio_data.shape}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate, format="mp3")
            logger.info(f"Audio file written: {tmp_file.name}, size={os.path.getsize(tmp_file.name)} bytes")
            result = recognize_song(audio_path=tmp_file.name)  # Llamada directa a la herramienta
        os.unlink(tmp_file.name)
        if "error" not in result:
            logger.info(f"Recognition successful: {result}")
            return f"🎵 **{result['Song']}** by {result['Artist']}"
        logger.info(f"Recognition failed: {result['error']}")
        return f"Recognition failed: {result['error']}"
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        return f"Error: {str(e)}"

# Interfaz Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Music Recognition with smolagents")
    audio_status = gr.Markdown("1. Record audio using the mic below\n2. Click 'Recognize Song'")
    audio_input = gr.Audio(label="Record Audio", type="numpy", interactive=True)
    record_btn = gr.Button("Recognize Song", variant="primary")
    output = gr.Markdown("Recognition result will appear here")

    record_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=output
    )

demo.launch(server_name="0.0.0.0", debug=True, show_error=True)
