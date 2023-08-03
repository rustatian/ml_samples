import whisper

model = whisper.load_model('tiny', download_root='/home/valery/.whisper/models')
audio = whisper.load_audio('./call.m4a')

options = {
    "language": "en", # input language, if omitted is auto detected
    "task": "transcribe" # or "transcribe" if you just want transcription
}
#options = whisper.DecodingOptions()#(fp16=False)
result = whisper.transcribe(model, audio, **options)
print(f"Decoded text: {result}")

import openai
from pydub import AudioSegment

audio = open("./call.m4a", "rb")
segment = AudioSegment.from_file(audio, "m4a")
ten_minutes = 10 * 60 * 1000
first_10_mins = segment[:ten_minutes]
first_10_mins.export("whisper-1.mp3", format="mp3")


# openai_audio = open("./whisper-1.mp3", "rb")
# openai.api_key=''
# transcript = openai.Audio.transcribe("whisper-1", openai_audio)
# print(transcript)