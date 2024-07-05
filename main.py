
import assemblyai as aai
from pytube import YouTube
from pydub import AudioSegment
import os

    
def video_to_audio(yt_url):
  yt = YouTube(yt_url)
  ys = yt.streams.filter(only_audio=True).first()
  ad = ys.download()
  base, ext = os.path.splitext(ad)
  audio = AudioSegment.from_file(ad)
  audio.export(base+'.mp3',format='mp3')
  os.remove(ad)
  print("Download Complete!")
  return base+'.mp3'

def audio_to_Text(filepath):
    aai.settings.api_key = "aded0399c8dc45e5b605a8856af99fd6"
    # URL of the file to transcribe
    
    # FILE_URL = "./audio1.mp3"

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filepath)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        return
    else:
        return(transcript.text)


# yt_url = input("enter the link of youtube video: ")
# audio_file_name = video_to_audio(yt_url)
# filepath = audio_file_name
filepath = "./a3.mp3"
text_from_video = audio_to_Text(filepath)
print(text_from_video)

