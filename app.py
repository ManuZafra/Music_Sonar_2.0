from smolagents import CodeAgent, HfApiModel, tool
import datetime
import requests
import yaml
import os
import json
import gradio as gr
from tools.final_answer import FinalAnswerTool
import logging
import tempfile
import soundfile as sf
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('music_recognition_app')

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

HISTORY_FILE = "song_history.json"
LANGUAGES = {"English": "en", "Español": "es", "Français": "fr"}

UI_MESSAGES = {
    "en": {
        "title": "Music Recognition & Fun Facts",
        "subtitle": "Identify songs in real-time and learn more",
        "rec_button": " START LISTENING",
        "please_record": "Click 'START LISTENING' to recognize songs in real-time",
        "recording": "Listening... Play music to recognize",
        "song_recognized": "Song recognized!",
        "about_artist": "About",
        "ask_more": "Ask me more about this artist or music",
        "chat_placeholder": "Ask about the song or artist...",
        "send_button": "Send"
    },
    "es": {
        "title": "Reconocimiento de Música y Datos Curiosos",
        "subtitle": "Identifica canciones en tiempo real y aprende más",
        "rec_button": " EMPEZAR A ESCUCHAR",
        "please_record": "Haz clic en 'EMPEZAR A ESCUCHAR' para reconocer canciones",
        "recording": "Escuchando... Reproduce música para reconocer",
        "song_recognized": "¡Canción reconocida!",
        "about_artist": "Sobre",
        "ask_more": "Pregúntame más sobre este artista o música",
        "chat_placeholder": "Pregunta sobre la canción o artista...",
        "send_button": "Enviar"
    },
    "fr": {
        "title": "Reconnaissance de Musique et Anecdotes",
        "subtitle": "Identifiez des chansons en temps réel et apprenez davantage",
        "rec_button": " COMMENCER À ÉCOUTER",
        "please_record": "Cliquez sur 'COMMENCER À ÉCOUTER' pour reconnaître",
        "recording": "Écoute... Jouez de la musique pour reconnaître",
        "song_recognized": "Chanson reconnue!",
        "about_artist": "À propos de",
        "ask_more": "Demandez-moi plus sur cet artiste ou la musique",
        "chat_placeholder": "Posez une question sur la chanson ou l'artiste...",
        "send_button": "Envoyer"
    }
}

@tool
def recognize_song(audio_path: str) -> dict:
    """Recognize a song from an audio file using the ACRCloud API.

    Args:
        audio_path: The file path to the audio file that will be sent to ACRCloud for recognition.

    Returns:
        dict: A dictionary containing song details (Song, Artist, Album, Recognition Date) or an error message if recognition fails.
    """
    logger.info(f"recognize_song docstring: {recognize_song.__doc__}")
    ACR_ACCESS_KEY = os.getenv("ACR_ACCESS_KEY")
    ACR_SECRET_KEY = os.getenv("ACR_SECRET_KEY")
    if not os.path.exists(audio_path):
        return {"error": "The audio file does not exist"}
    try:
        url = "http://identify-eu-west-1.acrcloud.com/v1/identify"
        data = {"access_key": ACR_ACCESS_KEY, "data_type": "audio", "sample_rate": 44100, "audio_format": "mp3"}
        with open(audio_path, 'rb') as file:
            files = {"sample": file}
            response = requests.post(url, data=data, files=files)
        if response.status_code != 200:
            return {"error": f"API Error: {response.status_code}"}
        result = response.json()
        if result['status']['code'] != 0:
            return {"error": result['status']['msg']}
        if 'metadata' not in result or 'music' not in result['metadata']:
            return {"error": "Could not recognize the song"}
        song_info = result['metadata']['music'][0]
        song_data = {
            "Song": song_info.get('title', 'Unknown'),
            "Artist": song_info.get('artists', [{}])[0].get('name', 'Unknown'),
            "Album": song_info.get('album', {}).get('name', 'Unknown'),
            "Recognition Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_to_history(song_data)
        return song_data
    except Exception as e:
        return {"error": f"Error processing audio: {str(e)}"}

def save_to_history(song_data):
    """Saves a song to the history file.

    Args:
        song_data: The song information to save.
    """
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []
        history.insert(0, song_data)
        if len(history) > 50:
            history = history[:50]
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")

@tool
def get_artist_info(artist_name: str, song_name: str = "", language: str = "en") -> str:
    """Get information about an artist and optionally a specific song.

    Args:
        artist_name: The name of the artist to get information about.
        song_name: The name of a specific song by the artist (optional, defaults to empty string).
        language: The language for the returned information (default: "en").

    Returns:
        str: A string containing information about the artist and song (if provided).
    """
    logger.info(f"get_artist_info docstring: {get_artist_info.__doc__}")
    prompts = {
        "en": f"Provide details about '{artist_name}'. Include biography, fun facts, and about '{song_name}' if available.",
        "es": f"Proporciona detalles sobre '{artist_name}'. Incluye biografía, datos curiosos y sobre '{song_name}' si está disponible.",
        "fr": f"Fournissez des détails sur '{artist_name}'. Incluez biographie, anecdotes et sur '{song_name}' si disponible."
    }
    language = language if language in prompts else "en"
    messages = [{"role": "user", "content": prompts[language]}]
    try:
        response = model(messages)
        return response.content
    except Exception as e:
        return f"Could not retrieve info: {str(e)}"

model = HfApiModel(max_tokens=2096, temperature=0.5, model_id='Qwen/Qwen2.5-Coder-32B-Instruct')
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(model=model, tools=[recognize_song, get_artist_info], max_steps=8, verbosity_level=1, prompt_templates=prompt_templates)

# Buffer para audio en tiempo real
audio_buffer = []
buffer_duration = 10  # 10 segundos por fragmento
sample_rate = 44100

def process_audio_stream(audio_chunk):
    """Processes audio chunks in real-time for song recognition.

    Args:
        audio_chunk: The audio data chunk received from FastRTC.

    Returns:
        str: The recognized song title and artist info, or a status message.
    """
    global audio_buffer
    if isinstance(audio_chunk, bytes):
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    else:
        audio_data = audio_chunk
    audio_buffer.extend(audio_data)
    if len(audio_buffer) >= sample_rate * buffer_duration:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            sf.write(tmp_file.name, np.array(audio_buffer), sample_rate, format="mp3")
            result = recognize_song(tmp_file.name)
        os.unlink(tmp_file.name)
        audio_buffer.clear()
        if "error" not in result:
            song_name = result['Song']
            artist_name = result['Artist']
            artist_info = get_artist_info(artist_name, song_name, "en")
            return f"🎵 **{song_name}** by {artist_name}\n\n{artist_info}"
        return "No match yet, keep playing..."
    return "Processing..."

stream = Stream(ReplyOnPause(process_audio_stream), modality="audio", mode="send-receive")

with gr.Blocks() as demo:
    lang_code = gr.State("en")
    audio_status = gr.State("no_audio")
    song_info_state = gr.State(None)
    artist_info_state = gr.State("")

    def get_ui_message(key, lang="en"):
        return UI_MESSAGES.get(lang, UI_MESSAGES["en"]).get(key, "")

    title_component = gr.Markdown(f"# 🎵 {get_ui_message('title', 'en')}")
    subtitle_component = gr.Markdown(get_ui_message('subtitle', 'en'))

    with gr.Row():
        language_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label=get_ui_message('choose_language', 'en'))

    with gr.Tab("Song Recognition"):
        audio_status_msg = gr.Markdown(f"*{get_ui_message('please_record', 'en')}*")
        with gr.Row():
            record_btn = gr.Button(get_ui_message('rec_button', 'en'), variant="primary")
        stream_output = gr.Markdown(label="Real-time Recognition")
        song_title_display = gr.Markdown("")
        artist_facts = gr.Markdown("")

    def toggle_audio_widget(lang_code):
        return "loading", get_ui_message('recording', lang_code), ""

    def update_ui_language(language_name):
        lang_code = LANGUAGES.get(language_name, "en")
        return (
            f"# 🎵 {get_ui_message('title', lang_code)}",
            get_ui_message('subtitle', lang_code),
            gr.update(label=get_ui_message('choose_language', lang_code)),
            gr.update(value=get_ui_message('rec_button', lang_code)),
            f"*{get_ui_message('please_record', lang_code)}*",
            lang_code
        )

    # Usar gr.Audio para capturar audio en tiempo real y procesarlo con stream
    audio_input = gr.Audio(source="microphone", streaming=True, visible=False)

    record_btn.click(
        fn=toggle_audio_widget,
        inputs=[lang_code],
        outputs=[audio_status, audio_status_msg, artist_facts]
    ).then(
        fn=process_audio_stream,  # Procesar directamente con la función de callback
        inputs=[audio_input],     # Audio recibido del micrófono
        outputs=[stream_output]
    ).then(
        fn=lambda output: (output.split('\n\n')[0], '\n\n'.join(output.split('\n\n')[1:]) if '\n\n' in output else ""),
        inputs=[stream_output],
        outputs=[song_title_display, artist_facts]
    )

    language_dropdown.change(
        fn=update_ui_language,
        inputs=[language_dropdown],
        outputs=[title_component, subtitle_component, language_dropdown, record_btn, audio_status_msg, lang_code]
    )

demo.launch(show_error=True, share=True, debug=True, server_name="0.0.0.0")
