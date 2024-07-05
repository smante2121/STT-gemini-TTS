import numpy as np
from flask import Flask, jsonify
from google.cloud import speech
from google.cloud import texttospeech
import sounddevice as sd
from six.moves import queue
import os
import google.generativeai as genai
from dotenv import load_dotenv



app = Flask(__name__)
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")



GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
'''
model = genai.GenerativeModel('gemini-1.5-flash',
                              system_instruction='You are getting information from a patient before they are '
                                                 'transferred to a nurse. Your conversation should be human like. '
                                                 'Greet the patient formally, ask for their'
                                                 'call back number, repeat the phone number and confirm it is correct'
                                                 'Ask if they are the patient, then ask for their date of birth. '
                                                 'Finally ask them for their symptoms.'
                                                 'After, thank the patient for their time and tell them they will be '
                                                 'transferred to a nurse.'
                                                 'Only use characters in your response. '
                                                 'If prompt received while responding, stop responding and only '
                                                 'respond to the new prompt.')
                                                '''

model = genai.GenerativeModel('gemini-1.5-flash',
                              system_instruction="Hello, let’s collect some information to expedite your call. What is your callback number? "
                                                 "I have your callback number as {number} [Slow down repeat]. Is that correct? "
                                                 "Are you the patient? "
                                                 "Great, could you please provide me with your date of birth? "
                                                 "Could you please provide the first three letters of your last name? "
                                                 "Got it. Are you a biological male or female? "
                                                 "What state are you in right now? "
                                                 "Perfect. In a few words, please tell me your main symptom or reason for the call today. "
                                                 "Give me a moment. We are all set.")
chat=model.start_chat(history=[])


# Audio recording parameters
RATE = 16000 # sample rate, how often the audio signal is sampled per second
CHUNK = int(RATE / 10)  # 100ms # how many samples are in each chunk of audio data

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self._closed = True # stream state

    def __enter__(self):
        self._closed = False # open the stream
        return self

    def __exit__(self, type, value, traceback):
        self._closed = True # close the stream

    def generator(self):
        # generate audio chunks from the mic
        with sd.InputStream(samplerate=self._rate, channels=1, dtype='int16', blocksize=self._chunk) as stream:
            while not self._closed:
                data, overflowed = stream.read(self._chunk) # read a chuck of audio
                if overflowed:
                    print("Audio buffer overflowed") # notify if buffer overflow
                self._buff.put(data.tobytes()) # put data into queue
                yield self._buff.get() # yield audio data from the queue

def listen_print_loop(responses):
    # Process and print the transcription results as they are received.
    num_chars_printed = 0 # Track the number of characters printed to handle overwrites
    buffer=""
    response_active=False
    for response in responses:
        if not response.results:
            continue

        result = response.results[0] # get the first result
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript # get the transcript
        if result.is_final: # check if the result is final
            buffer+=transcript # add the transcript to the buffer
            print(buffer)
            response=chat.send_message(buffer) # send the buffer to the chat model
            print(response.text) # print the response from the chat model
            synthesize_text(response.text) # synthesize the response from the chat model
            buffer="" # reset the buffer
            num_chars_printed = 0 # reset the number of characters printed
        else:
            overwrite_chars=' ' * (num_chars_printed - len(transcript)) # calculate the number of chars to overwrite
            print(transcript + overwrite_chars, end='\r') # print the transcript
            num_chars_printed = len(transcript) # update the number of characters printed


@app.route('/')
def index():
    return jsonify({"message": "API is running. Use /transcribe to start the transcription."})


@app.route('/transcribe')
def transcribe_stream():
    # Endpoint to start transcribing audio from the microphone
    client = speech.SpeechClient() # Initialize the Google Cloud Speech client

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US", # Set the audio encoding, sample rate, and language
        enable_automatic_punctuation=True,
        model="telephony",
        speech_contexts=[speech.SpeechContext(phrases=["specific term1", "specific term2"], boost=20.0)]
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator if content)

        responses = client.streaming_recognize(streaming_config, requests)

        try:
            listen_print_loop(responses)
        except Exception as e:
            print(f"Error: {e}")

    return jsonify({"status": "transcription completed"})



def synthesize_text(text):
    """Synthesizes speech from the input string of text."""

    client = texttospeech.TextToSpeechClient() # Initialize the Google Cloud Text-to-Speech client
    input_text = texttospeech.SynthesisInput(text=text) # set the input text


    voice = texttospeech.VoiceSelectionParams( # set the voice parameters
        language_code="en-US", # set language as english
        name="en-US-Standard-I",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, # set the audio encoding
    )

    response = client.synthesize_speech( # synthesize the speech
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    audio_data=np.frombuffer(response.audio_content, dtype=np.int16)
    sd.play(audio_data, samplerate=24000)
    sd.wait()



if __name__ == '__main__':
    app.run(debug=True, ssl_context=('server.crt', 'server.key'))