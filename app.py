import re
import sys
import time
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

model = genai.GenerativeModel('gemini-1.5-flash',
                              system_instruction='Respond as if you are a human having a conversation over the phone. '
                                                 'Only use characters in your response. '
                                                 'Keep responses less than 70 characters.'
                                                 'After receiving prompt, wait for 3 second before responding.'
                                                 'If prompt received while responding, stop responding and only '
                                                 'respond to the new prompt.')
chat=model.start_chat(history=[])


# Audio recording parameters
RATE = 16000 # sample rate, how often the audio signal is sampled per second
CHUNK = int(RATE / 10)  # 100ms # how many samples are in each chunk of audio data


class Conversation:
    def __init__(self):
        self.step=0
        self.responses={}
        self.questions = [
            "Hello, let’s collect some information to expedite your call. What is your callback number?",
            "Please confirm, is that correct?",
            "Are you the patient?",
            "Could you please provide me with your date of birth?",
            "Could you please provide the first three letters of your last name?",
            "Got it. Are you a biological male or female?",
            "What state are you in right now?",
            "In a few words, please tell me your main symptom or reason for the call today.",
            "Give me a moment. We are all set."
        ]
    def process_response(self, user_input):

        if user_input.strip()=="":
            return 'Please provide a valid response.'

        if self.step == 1:  # Handle confirmation step specifically
            if "yes" in user_input.lower() or "correct" in user_input.lower():
                self.step += 1  # Move to next question if confirmed
            else:
                return self.questions[self.step]  # Ask again if not confirmed

        self.responses[self.step] = user_input  # Store user input for current step
        self.step += 1  # Move to the next step

        if self.step >= len(self.questions):
            return "Thank you, your information has been processed."
        return self.questions[self.step]

    def get_initial_question(self):
        return self.questions[self.step]





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
                if data is not None and not overflowed:
                    self._buff.put(data.tobytes())
                    yield self._buff.get()
                else:
                    print("Audio buffer overflowed")


def listen_print_loop(responses):
    # Process and print the transcription results as they are received.
    num_chars_printed = 0  # Track the number of characters printed to handle overwrites
    buffer = ""
    response_active = False

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]  # get the first result
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript  # get the transcript
        if not result.is_final:
            # Update the output with the partial transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            # When the result is final, clear the current line and print the full transcript
            overwrite_chars = ' ' * num_chars_printed
            print(transcript + overwrite_chars)  # Clear the output with spaces
            num_chars_printed = 0  # Reset the character count after the final print

            buffer += transcript.strip()  # Add the final transcript to the buffer
            if buffer:  # Check if the buffer is not empty
                print("Transcript:", buffer)
                next_question = conversation.process_response(buffer)
                print('Next question:', next_question)
                synthesize_text(next_question)  # Synthesize the question based on the response
                buffer = ""  # Reset the buffer after processing
                time.sleep(1.5)
            else:
                print("No valid transcript detected.")





@app.route('/transcribe')
def transcribe_stream():
    global conversation
    conversation=Conversation() # create an instance of the Conversation class
    # Endpoint to start transcribing audio from the microphone
    client = speech.SpeechClient() # Initialize the Google Cloud Speech client

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US", # Set the audio encoding, sample rate, and language
        enable_automatic_punctuation=True,
        model="telephony",
        use_enhanced=True,
        speech_contexts=[speech.SpeechContext(phrases=["yes", "no", "correct", "repeat"], boost=15)],
        #speech_contexts=[speech.SpeechContext(phrases=["specific term1", "specific term2"], boost=20.0)],
        enable_word_time_offsets=True  # Helpful for debugging timings
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
        print("Starting conversation with initial question...")
        initial_question = conversation.get_initial_question()
        synthesize_text(initial_question)
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