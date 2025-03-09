import os
import requests
import gradio as gr
import logging
import tempfile
import soundfile as sf
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("music_recognition_app")

# Verificar FastRTC
try:
    from fastrtc import Stream, ReplyOnPause
    logger.info("FastRTC imported successfully")
except ImportError:
    logger.error("FastRTC not installed")
    raise

# Función de reconocimiento con ACRCloud
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
        data = {"access_key": ACR_ACCESS_KEY, "data_type": "audio", "sample_rate": 44100, "audio_format": "mp3"}
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

# Buffer de audio
audio_buffer = []
buffer_duration = 5  # 5 segundos
sample_rate = 44100

def process_audio_stream(audio_data):
    global audio_buffer
    logger.info(f"Received audio data: type={type(audio_data)}, length={len(audio_data) if audio_data is not None else 'None'}")
    if audio_data is None:
        logger.info("No audio data received")
        return "Processing..."
    try:
        audio_array = np.array(audio_data)
        logger.info(f"Audio array: shape={audio_array.shape}, min={audio_array.min()}, max={audio_array.max()}")
        audio_buffer.extend(audio_data)
        logger.info(f"Buffer size: {len(audio_buffer)}")
        if len(audio_buffer) >= sample_rate * buffer_duration:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                sf.write(tmp_file.name, np.array(audio_buffer), sample_rate, format="mp3")
                logger.info(f"Audio file written: {tmp_file.name}, size={os.path.getsize(tmp_file.name)} bytes")
                result = recognize_song(tmp_file.name)
            os.unlink(tmp_file.name)
            audio_buffer.clear()
            if "error" not in result:
                return f"🎵 **{result['Song']}** by {result['Artist']}"
            return f"Recognition failed: {result['error']}"
        return "Processing..."
    except Exception as e:
        logger.error(f"Error in process_audio_stream: {str(e)}")
        return f"Error: {str(e)}"

stream = Stream(ReplyOnPause(process_audio_stream), modality="audio", mode="send-receive")

# Interfaz de Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Music Recognition")
    audio_status = gr.Markdown("Click 'START LISTENING' to begin")
    record_btn = gr.Button("START LISTENING", variant="primary")
    output = gr.Markdown("Recognition result will appear here")
    audio_input = gr.Audio(streaming=True, interactive=True, label="Mic Input")

    def start_listening():
        return "Listening..."

    record_btn.click(
        fn=start_listening,
        inputs=None,
        outputs=audio_status
    ).then(
        fn=process_audio_stream,
        inputs=audio_input,
        outputs=output
    )

demo.launch(show_error=True, debug=True, server_name="0.0.0.0")
