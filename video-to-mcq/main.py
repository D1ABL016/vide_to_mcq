# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai


aai.settings.api_key = "aded0399c8dc45e5b605a8856af99fd6"

# URL of the file to transcribe
FILE_URL = "./audio1.mp3"


transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    print(transcript.text)
