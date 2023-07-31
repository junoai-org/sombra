import io
import os
import speech_recognition as sr
import whisperx
import torch
import openai
from dotenv import load_dotenv
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from threading import Thread
import eel

load_dotenv()   # Cargar variables de entorno

## Variables

openai.api_key = os.getenv("OPENAI_API_KEY")
batch_size = int(os.getenv("BATCH_SIZE"))
model = os.getenv("MODEL")
language = os.getenv("LANGUAGE")
chatgpt_model = os.getenv("CHATGPT_MODEL")

eel.init('web')
# Variables globales
phrase_time = None
last_sample = bytes()
data_queue = Queue()
transcription = ['']
transcribing = False


@eel.expose
def get_updates():
    global transcription
    transcription_text = '\n'.join(transcription)
    return transcription_text

@eel.expose
def toggle_recording():
    global transcribing
    value = transcribing
    if transcribing:
        stop_transcription()
    else:
        start_transcription()

    return not value

@eel.expose
def execute(context,task):
    print(context,task)
    global transcription
    transcription_text = '\n'.join(transcription)
    response = openai.ChatCompletion.create(
        model=chatgpt_model,
        messages=[
            {
                "role": "system",
                "content": f"You are a AI assistant that help to process the transcription of a meeting.\n{context}"
            },
            {
                "role": "user",
                "content": transcription_text
            },
            {
                "role": "user",
                "content": task
            }
        ],
        stream=True,
    )

    collected_messages = []
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        eel.set_result(''.join([m.get('content', '') for m in collected_messages]))
    
    return ''.join([m.get('content', '') for m in collected_messages])

@eel.expose
def clear():
    global transcription
    transcription = ['']
    return True

def start_transcription(
    model="medium",
    non_english=False,
    energy_threshold=500,
    record_timeout=2,
    phrase_timeout=2.5,
    default_microphone='pulse',
):
    global transcribing
    if transcribing:
        print("Ya se está realizando una transcripción.")
        return

    transcribing = True

    if 'linux' in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        if not transcribing:
            return

        data = audio.get_raw_data()
        data_queue.put(data)

    run_thread = Thread(target=_main_loop, args=(recorder, source, record_callback, record_timeout, phrase_timeout))
    run_thread.start()

def _main_loop(recorder, source, record_callback, record_timeout, phrase_timeout):
    global transcription
    global phrase_time
    global last_sample

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Starting transcription...")

    temp_file = NamedTemporaryFile().name
    while transcribing:
        try:
            now = datetime.utcnow()

            if not data_queue.empty():
                phrase_complete = False

                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True

                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                    
                audio = whisperx.load_audio(temp_file)
                result = audio_model.transcribe(audio, batch_size=batch_size)
                text = ""
                for segment in result['segments']:
                    text += segment['text']

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                
                os.system('cls' if os.name=='nt' else 'clear')
                with open("out.txt", 'w') as f:
                    for line in transcription:
                        print(line)
                        f.write(line+"\n")

                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription complete.\n")
    for line in transcription:
        print(line)

def stop_transcription():
    global transcribing
    transcribing = False
    print("Stopping transcription...")

# Cargar el modelo globalmente
audio_model = whisperx.load_model(model, device="cuda" if torch.cuda.is_available() else "cpu", language=language)

if __name__ == "__main__":
    # Prueba las funciones start_transcription y stop_transcription
    eel.start('index.html', size=(1000, 1000))