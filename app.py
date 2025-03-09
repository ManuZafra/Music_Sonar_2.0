import os
import requests
import gradio as gr
import logging
import tempfile
import soundfile as sf
import numpy as np
import hmac
import hashlib
import time
import base64

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("music_recognition_app")

# ACRCloud con firma
def recognize_song(audio_path: str) -> dict:
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
        timestamp = str(int(time.time()))  # Tiempo actual en segundos
        signature_version = "1"
        method = "POST"

        # Generar la firma
        string_to_sign = f"{method}\n/v1/identify\n{ACR_ACCESS_KEY}\n{signature_version}\n{timestamp}"
        sign = hmac.new(
            ACR_SECRET_KEY.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha1
        ).digest()
        signature = base64.b64encode(sign).decode('utf-8')

        # Parámetros de la solicitud
        data = {
            "access_key": ACR_ACCESS_KEY,
            "sample_rate": "44100",
            "audio_format": "mp3",
            "signature_version": signature_version,
            "signature": signature,
            "timestamp": timestamp,
            "data_type": "audio"
        }

        with open(audio_path, "rb") as file:
            files = {"sample": file}
            response = requests.post(url, data=data, files=files)

        logger.info(f"ACRCloud response: status={response.status_code}, content={response.text}")
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        result = response.json()
        if result.get("status", {}).get("code", -1) != 0:
            return {"error": result["status"]["msg"]}
        if "metadata" not in result or "music" not in result["metadata"]:
            return {"error": "Could not recognize the song"}
        song_info = result["metadata"]["music"][0]
        return {
            "Song": song_info.get("title", "Unknown"),
            "Artist": song_info.get("artists", [{}])[0].get("name", "Unknown"),
        }
    except Exception as e:
        logger.error(f"Error in recognize_song: {str(e)}")
        return {"error": f"Error: {str(e)}"}

def process_audio(audio):
    logger.info(f"Received audio: {audio}")
    if audio is None:
        logger.info("No audio received")
        return "Please record some audio"
    try:
        sample_rate, audio_data = audio
        logger.info(f"Audio data: sample_rate={sample_rate}, shape={audio_data.shape}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate, format="mp3")
            logger.info(f"Audio file written: {tmp_file.name}, size={os.path.getsize(tmp_file.name)} bytes")
            result = recognize_song(tmp_file.name)
        os.unlink(tmp_file.name)
        if "error" not in result:
            return f"🎵 **{result['Song']}** by {result['Artist']}"
        return f"Recognition failed: {result['error']}"
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        return f"Error: {str(e)}"

# Interfaz
with gr.Blocks() as demo:
    gr.Markdown("# Music Recognition")
    audio_status = gr.Markdown("Click 'Record and Recognize' to capture audio")
    audio_input = gr.Audio(label="Record Audio", type="numpy", interactive=True)
    record_btn = gr.Button("Record and Recognize", variant="primary")
    output = gr.Markdown("Recognition result will appear here")

    record_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=output
    )

demo.launch(show_error=True, debug=True, server_name="0.0.0.0")
