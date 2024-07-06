import numpy as np
from flask import Flask, render_template, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from google.cloud import speech
from google.cloud import texttospeech
import sounddevice as sd
from six.moves import queue
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re
from forms import LookUpForm
import logging
from threading import Thread, Lock
import time

app = Flask(__name__)
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
db = SQLAlchemy(app)
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
app.config['PROPAGATE_EXCEPTIONS'] = True
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash',
                              system_instruction="Ask each of these questions: Hello, let’s collect some information to expedite your call. What is your callback number? After they give you the call back number ask the next question. Ensure that they give you a nine digit number."
                                                 "I have your callback number as {number}. Is that correct? "
                                                 "Are you the patient? "
                                                 "Great, could you please provide me with your date of birth? "
                                                 "Could you please provide the first three letters of your last name? "
                                                 "Got it. Are you a biological male or female? "
                                                 "What state are you in right now? "
                                                 "Perfect. In a few words, please tell me your main symptom or reason for the call today. "
                                                 "Give me a moment. We are all set.")
chat=model.start_chat(history=[])

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)
lock=Lock()
global_stream = None

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self._closed = True

    def __enter__(self):
        self._closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._closed = True

    def start(self):
        print("Stream started")
        self._closed = False

    def stop(self):
        print("Stream has been closed")
        self._closed = True

    def generator(self):
        # generate audio chunks from the mic
        with sd.InputStream(samplerate=self._rate, channels=1, dtype='int16', blocksize=self._chunk) as stream:
            while not self._closed:
                data, overflowed = stream.read(self._chunk) # read a chuck of audio
                if overflowed:
                    print("Audio buffer overflowed") # notify if buffer overflow
                if data is not None and len(data) > 0:
                    self._buff.put(data.tobytes()) # put data into queue
                    yield self._buff.get() # yield audio data from the queue
                else:
                    print("No audio data") # notify if no audio data

def extract_callback_number(buffer):
    match = re.search(r'\b(\d{3}[- ]?\d{3}[- ]?\d{4})\b', buffer)
    if match:
        return re.sub(r'[- ]', '', match.group(1))
    return None

def extract_is_patient(buffer):
    # Searching for direct affirmations or negations following typical questions about the caller's identity
    patterns = [
        r'are you the patient\?\s*(yes|no)',
        r'is this call for yourself\?\s*(yes|no)'
    ]
    for pattern in patterns:
        match = re.search(pattern, buffer, re.IGNORECASE)
        if match:
            # Return True if 'yes', False if 'no'
            return True if match.group(1).lower() == 'yes' else False
    return None  # Return None if no clear answer is found

def extract_date_of_birth(buffer):
    month_to_number = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04',
        'May': '05', 'June': '06', 'July': '07', 'August': '08',
        'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }

    # Handle numerical date formats
    match = re.search(r'\b(\d{1,2})[/-]?(\d{1,2})[/-]?(\d{4})\b', buffer)
    if match:
        return '/'.join(match.groups())

    # Handle month-day-year with possible ordinal suffixes and optional comma
    match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(st|nd|rd|th)?\s*(,)?\s*(\d{4})\b', buffer, re.IGNORECASE)
    if match:
        month, day, ordinal_suffix, comma, year = match.groups()
        month_number = month_to_number[month.capitalize()]
        return f"{month_number}/{day.zfill(2)}/{year}"

    # Handle more generic month-day-year formats without explicit separators
    pattern = r'\b(' + '|'.join(month_to_number.keys()) + r')\s+(\d{1,2})\s+(\d{4})\b'
    match = re.search(pattern, buffer, re.IGNORECASE)
    if match:
        month, day, year = match.groups()
        month_number = month_to_number[month.capitalize()]
        return f"{month_number}/{day.zfill(2)}/{year}"

    # Handle compact numeric dates (e.g., 6212003)
    match = re.search(r'\b(\d{1})(\d{2})(\d{4})\b', buffer)
    if match:
        month, day, year = match.groups()
        return f"{month.zfill(2)}/{day.zfill(2)}/{year}"

    return None

def extract_last_name_letters(buffer):
    # Capture spaced out letters and typical last name entries
    match = re.search(r'\b(?:the first three letters of your last name are |last name\? )([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', buffer, re.IGNORECASE)
    if match:
        return ''.join(match.groups()).capitalize()
    # Handle a more typical last name entry if three-letter format not used
    match = re.search(r'\bmy last name is (\w{3})', buffer, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

def extract_gender(buffer):
    match = re.search(r'\b(male|female)\b', buffer, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_state(buffer):
    states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
              "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
              "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
              "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
              "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
              "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
              "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
    for state in states:
        if re.search(r'\b' + re.escape(state) + r'\b', buffer, re.IGNORECASE):
            return state
    return None

def extract_symptom(buffer):
    # Enhance symptom capture by ignoring the echo of the question
    pattern = r'(?<=tell me your main symptom or reason for the call today\.\s*)[A-Za-z].*?([^.]+)\.'
    match = re.search(pattern, buffer, re.IGNORECASE | re.DOTALL)
    if match:
        # Clean up response from any lead text echoed from the question
        response = match.group(1).strip()
        return response.replace('I\'m calling because ', '').replace('My symptom is ', '')
    return None

def listen_print_loop(responses, stream):
    num_chars_printed = 0
    buffer = ""
    full_transcript = ""

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        if transcript:
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))
            print(transcript + overwrite_chars, end='\r')
            num_chars_printed = len(transcript)

            if result.is_final:
                full_transcript += transcript + " "
                buffer += transcript
                print('\n' + buffer)

                if buffer.strip():
                    response = chat.send_message(buffer)
                    print(response.text)
                    synthesize_text(response.text)
                buffer = ""
                num_chars_printed = 0

    return full_transcript

def process_full_transcript(full_transcript):
    details = {
        "number": extract_callback_number(full_transcript),
        "patient": extract_is_patient(full_transcript),
        "dob": extract_date_of_birth(full_transcript),
        "lastName": extract_last_name_letters(full_transcript),
        "gender": extract_gender(full_transcript),
        "state": extract_state(full_transcript),
        "symptom": extract_symptom(full_transcript)
    }

    if any(details.values()):
        return details
    else:
        print("No valid data extracted from the full transcript.")
        return None

def save_to_database(details):
    try:
        new_entry = phone(**details)
        db.session.add(new_entry)
        db.session.commit()
        logging.info(f"Saved to database: {new_entry}")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error saving to database: {e}")




@app.route('/transcribe/start', methods=['POST'])
def start_transcription():
    global global_stream
    if global_stream is None or global_stream._closed:
        global_stream = MicrophoneStream(RATE, CHUNK)
        global_stream.start()

        thread = Thread(target=process_stream)
        thread.start()
    return jsonify({"status": "transcription started"})

def process_stream():
    global global_stream
    with app.app_context():  # Ensures the use of Flask's application context
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="telephony"
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
        if global_stream is None:
            logging.error("Stream is not active.")
            return None
        with global_stream as stream:
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator if content and not content.isspace())
            responses = client.streaming_recognize(streaming_config, requests)
            try:
                full_transcript = listen_print_loop(responses, stream)
                if full_transcript:
                    details = process_full_transcript(full_transcript)
                    if details:
                        save_to_database(details)
                        print("Transcription stopped, data processed and saved.")
                    else:
                        print("No valid data extracted.")
                else:
                    print("No transcript was processed.")
            except Exception as e:
                logging.error(f"Failed to process stream: {e}")

@app.route('/transcribe/stop', methods=['POST'])
def stop_transcription():
    global global_stream
    with lock:
        if global_stream and not global_stream._closed:
            global_stream.stop()
            try:
                # Wait for a short period to ensure the last bits of audio are processed.
                time.sleep(1)
                full_transcript = process_stream()
                if full_transcript:
                    details = process_full_transcript(full_transcript)
                    if details:
                        try:
                            save_to_database(details)
                            response_message = "Transcription stopped, data processed and saved."
                        except Exception as e:
                            db.session.rollback()
                            logging.error(f"Database save error: {e}")
                            response_message = "Transcription stopped, data processing succeeded but save failed."
                    else:
                        response_message = "Transcription stopped, no valid data extracted."
                else:
                    response_message = "Transcription stopped, but no transcript was processed."
            except Exception as e:
                logging.error(f"Error processing stream: {e}")
                response_message = "Error stopping transcription and processing data."
            return jsonify({"status": response_message})
        else:
            return jsonify({"status": "Transcription not active or already stopped"})


@app.route('/transcribe/start', methods=['GET', 'POST'])
def transcribe_stream():
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="telephony",
        speech_contexts=[speech.SpeechContext(phrases=["specific term1", "specific term2"], boost=20.0)]
    )

    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator if content)

        responses = client.streaming_recognize(streaming_config, requests)
        if listen_print_loop(responses, stream):
            print("Transcription process completed and stopped.")
            return redirect(url_for('lookup_number'))
        else:
            print("Transcription did not meet stop criteria.")

    return jsonify({"status": "transcription ended"})

def synthesize_text(text):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-I",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    audio_data = np.frombuffer(response.audio_content, dtype=np.int16)
    sd.play(audio_data, samplerate=24000)
    sd.wait()

class phone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(15), index=True, unique=True)
    patient = db.Column(db.String(100), index=True, unique=False)
    dob = db.Column(db.String(100), index=True, unique=False)
    lastName = db.Column(db.String(100), index=True, unique=False)
    gender = db.Column(db.String(100), index=True, unique=False)
    state = db.Column(db.String(100), index=True, unique=False)
    symptom = db.Column(db.String(100), index=True, unique=False)

    def __repr__(self):
        return f"Phone number: {self.number} Name: {self.patient} DOB: {self.dob} Last Name: {self.lastName} State: {self.state} Symptom: {self.symptom}"

@app.route('/', methods=["GET", "POST"])
def lookup_number():
    form = LookUpForm(csrf_enabled=False)
    if form.validate_on_submit():
        number = form.number.data
        phone_number = phone.query.filter_by(number=number).first()
        if phone_number:
            return redirect(url_for('patient_info', number=number))
    return render_template('lookup.html', form=form)

@app.route('/<number>')
def patient_info(number):
    formatted_number = number.replace('-', '').replace(' ', '')
    phone_number = phone.query.filter_by(number=formatted_number).first_or_404()
    return render_template('patient_info.html', phone_number=phone_number)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, ssl_context=('server.crt', 'server.key'), threaded=True)