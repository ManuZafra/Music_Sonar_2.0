from smolagents import CodeAgent, HfApiModel, tool
import datetime
import requests
import yaml
import os
import json
import gradio as gr
import logging
import tempfile
import soundfile as sf
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("music_recognition_app")

# Verificar versión de smolagents
import smolagents
logger.info(f"smolagents version: {smolagents.__version__}")

# Verificación de FastRTC con VAD
try:
    from fastrtc import Stream, ReplyOnPause
    ReplyOnPause(lambda x: x)
    logger.info("FastRTC with VAD support is installed correctly")
except ImportError:
    logger.error("FastRTC not installed")
    raise
except RuntimeError as e:
    logger.error(f"FastRTC VAD support missing: {str(e)}")
    raise

# Variables globales
HISTORY_FILE = "song_history.json"
LANGUAGES = {"English": "en", "Español": "es", "Français": "fr"}

@tool
def recognize_song(audio_path: str) -> dict:
    """
    Recognize a song from an audio file using the ACRCloud API.

    Args:
        audio_path: The file path to the audio file that will be sent to ACRCloud for recognition.

    Returns:
        dict: A dictionary containing song details (Song, Artist, Album, Recognition Date) or an error message.
    """
    logger.info(f"recognize_song docstring: {recognize_song.__doc__}")

    ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
    ACR_SECRET_KEY = os.getenv("ACR_SECRET_KEY")

    if not os.path.exists(audio_path):
        return {"error": "The audio file does not exist"}

    try:
        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        data = {"access_key": ACR_ACCESS_KEY, "data_type": "audio", "sample_rate": 44100, "audio_format": "mp3"}

        with open(audio_path, "rb") as file:
            files = {"sample": file}
            response = requests.post(url, data=data, files=files)

        logger.info(f"API Response: {response.status_code} - {response.text}")

        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}

        result = response.json()

        if result.get("status", {}).get("code", -1) != 0:
            return {"error": result['status']['msg']}

        if "metadata" not in result or "music" not in result["metadata"]:
            return {"error": "Could not recognize the song"}

        song_info = result["metadata"]["music"][0]
        song_data = {
            "Song": song_info.get("title", "Unknown"),
            "Artist": song_info.get("artists", [{}])[0].get("name", "Unknown"),
            "Album": song_info.get("album", {}).get("name", "Unknown"),
            "Recognition Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return song_data

    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}

# Buffer de audio en tiempo real
audio_buffer = []
buffer_duration = 10  # 10 segundos por fragmento
sample_rate = 44100

def process_audio_stream(audio_data):
    """
    Processes audio chunks in real-time for song recognition.

    Args:
        audio_data: The audio data chunk received from the microphone.

    Returns:
        str: The recognized song title and artist info, or a status message.
    """
    global audio_buffer
    if audio_data is None:
        return "Processing..."

    audio_buffer.extend(audio_data)
    logger.info(f"Audio buffer size: {len(audio_buffer)}")

    if len(audio_buffer) >= sample_rate * buffer_duration:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            sf.write(tmp_file.name, np.array(audio_buffer), sample_rate, format="mp3")
            logger.info(f"Archivo de audio guardado en: {tmp_file.name}")

            result = recognize_song(tmp_file.name)

        os.unlink(tmp_file.name)
        audio_buffer.clear()

        if "error" not in result:
            song_name = result["Song"]
            artist_name = result["Artist"]
            return f"🎵 **{song_name}** by {artist_name}"

        return "No match yet, keep playing..."

    return "Processing..."

stream = Stream(ReplyOnPause(process_audio_stream), modality="audio", mode="send-receive")

with gr.Blocks() as demo:
    lang_code = gr.State("en")
    audio_status = gr.State("no_audio")
    song_info_state = gr.State(None)

    def get_ui_message(key, lang="en"):
        messages = {
            "title": "Music Recognition & Fun Facts",
            "subtitle": "Identify songs in real-time and learn more",
            "rec_button": "START LISTENING",
            "please_record": "Click 'START LISTENING' to recognize songs",
            "recording": "Listening... Play music to recognize",
        }
        return messages.get(key, "")

    title_component = gr.Markdown(f"# 🎵 {get_ui_message('title', 'en')}")
    subtitle_component = gr.Markdown(get_ui_message('subtitle', 'en'))

    with gr.Row():
        language_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label="Language")

    with gr.Tab("Song Recognition"):
        audio_status_msg = gr.Markdown(f"*{get_ui_message('please_record', 'en')}*")
        with gr.Row():
            record_btn = gr.Button(get_ui_message("rec_button", "en"), variant="primary")
        stream_output = gr.Markdown(label="Real-time Recognition")

    def toggle_audio_widget(lang_code):
        return "loading", get_ui_message("recording", lang_code)

    def update_ui_language(language_name):
        lang_code = LANGUAGES.get(language_name, "en")
        return (
            f"# 🎵 {get_ui_message('title', lang_code)}",
            get_ui_message("subtitle", lang_code),
            gr.update(value=get_ui_message("rec_button", lang_code)),
            f"*{get_ui_message('please_record', lang_code)}*",
            lang_code,
        )

    record_btn.click(
        fn=toggle_audio_widget,
        inputs=[lang_code],
        outputs=[audio_status, audio_status_msg]
    ).then(
        fn=process_audio_stream,
        inputs=[gr.Audio(streaming=True)],
        outputs=[stream_output]
    )

    language_dropdown.change(
        fn=update_ui_language,
        inputs=[language_dropdown],
        outputs=[title_component, subtitle_component, record_btn, audio_status_msg, lang_code]
    )

demo.launch(show_error=True, share=False, debug=True, server_name="0.0.0.0")
