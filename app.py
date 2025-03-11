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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("music_recognition_app")

def recognize_song(audio_path: str) -> dict:
    logger.info(f"Starting recognition for: {audio_path}")
    ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
    ACR_SECRET_KEY = os.getenv("ACR_SECRET_KEY")
    logger.info(f"Using ACR_ACCESS_KEY: {ACR_ACCESS_KEY[:4]}...{ACR_ACCESS_KEY[-4:]}")
    logger.info(f"Using ACR_SECRET_KEY: {ACR_SECRET_KEY[:4]}...{ACR_SECRET_KEY[-4:]}")
    if not ACR_ACCESS_KEY or not ACR_SECRET_KEY:
        logger.error("ACRCloud credentials missing")
        return {"error": "ACRCloud credentials not set"}
    if not os.path.exists(audio_path):
        logger.error("Audio file does not exist")
        return {"error": "Audio file does not exist"}

    try:
        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        timestamp = str(int(time.time()))
        signature_version = "1"
        method = "POST"
        sample_bytes = str(os.path.getsize(audio_path))

        string_to_sign = f"{method}\n/v1/identify\n{ACR_ACCESS_KEY}\n{signature_version}\n{timestamp}"
        logger.info(f"String to sign: {repr(string_to_sign)}")
        sign = hmac.new(
            ACR_SECRET_KEY.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            hashlib.sha1
        ).digest()
        signature = base64.b64encode(sign).decode('utf-8')

        data = {
            "access_key": ACR_ACCESS_KEY,
            "sample_rate": "44100",
            "audio_format": "mp3",
            "signature_version": signature_version,
            "signature": signature,
            "timestamp": timestamp,
            "data_type": "audio",
            "sample_bytes": sample_bytes
        }
        logger.info(f"Request data: {data}")

        with open(audio_path, "rb") as file:
            files = {"sample": (os.path.basename(audio_path), file, "audio/mpeg")}
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
    logger.info(f"Processing audio: {audio}")
    if audio is None:
        logger.info("No audio received")
        return "Please record some audio first"
    try:
        sample_rate, audio_data = audio
        logger.info(f"Audio data: sample_rate={sample_rate}, shape={audio_data.shape}")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate, format="mp3")
            logger.info(f"Audio file written: {tmp_file.name}, size={os.path.getsize(tmp_file.name)} bytes")
            result = recognize_song(tmp_file.name)
        os.unlink(tmp_file.name)
        if "error" not in result:
            logger.info(f"Recognition successful: {result}")
            return f"🎵 **{result['Song']}** by {result['Artist']}"
        logger.info(f"Recognition failed: {result['error']}")
        return f"Recognition failed: {result['error']}"
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        return f"Error: {str(e)}"

# Interfaz simplificada para pruebas
demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(label="Record Audio", type="numpy"),
    outputs="text",
    title="Music Recognition",
    description="Record 5-10 seconds of a song and click Submit to recognize it."
)
demo.launch(server_name="0.0.0.0", debug=True, show_error=True)
