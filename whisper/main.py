import whisper

model = whisper.load_model('large', download_root='/home/valery/.whisper/models')
audio = whisper.load_audio('/home/valery/Downloads/7m.mp3')

options = {
    "language": "pl", # input language, if omitted is auto detected
    "task": "transcribe" # or "transcribe" if you just want transcription
}
#options = whisper.DecodingOptions()#(fp16=False)
result = whisper.transcribe(model, audio, **options)
f = open("result.txt", "w")
f.write(result["text"])
f.close()
# print(f"Decoded text: {result}")

# import openai
# from pydub import AudioSegment

# audio = open("./call.m4a", "rb")
# segment = AudioSegment.from_file(audio, "m4a")
# ten_minutes = 10 * 60 * 1000
# first_10_mins = segment[:ten_minutes]
# first_10_mins.export("whisper-1.mp3", format="mp3")


# openai_audio = open("./whisper-1.mp3", "rb")
# openai.api_key=''
# transcript = openai.Audio.transcribe("whisper-1", openai_audio)
# print(transcript)
