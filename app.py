import numpy as np
from flask import Flask, render_template, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from google.cloud import speech
from google.cloud import texttospeech
import sounddevice as sd
from six.moves import queue
from database import phone, save_to_database, db
import os
import google.generativeai as genai
from dotenv import load_dotenv
from forms import LookUpForm
import logging
from stream import listen_print_loop, synthesize_text, process_full_transcript, process_stream
from threading import Thread, Lock
import time
from MicrophoneStream import MicrophoneStream
from extraction import extract_callback_number, extract_is_patient, extract_date_of_birth, extract_last_name_letters, extract_gender, extract_state, extract_symptom

app = Flask(__name__)
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myDB.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
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




@app.route('/transcribe/start', methods=['POST']) # This is the transcription start route
def start_transcription(): # This function will be called when the URL is of the form '/transcribe/start'
    global global_stream
    if global_stream is None or global_stream._closed:
        global_stream = MicrophoneStream(RATE, CHUNK) # Create a MicrophoneStream object
        global_stream.start() # Start the stream
        thread = Thread(target=process_stream)
        thread.start()
    return jsonify({"status": "transcription started"})





@app.route('/transcribe/stop', methods=['POST']) # This is the transcription stop route
def stop_transcription(): # This function will be called when the URL is of the form '/transcribe/stop'
    global global_stream
    with lock: # Ensure only one thread can access the global_stream at a time
        if global_stream and not global_stream._closed:
            global_stream.stop()
            try:
                # Wait for a short period to ensure the last bits of audio are processed.
                time.sleep(1)
                full_transcript = process_stream() # Call the process_stream function
                if full_transcript:
                    details = process_full_transcript(full_transcript) # Call the process_full_transcript function
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




@app.route('/transcribe/start', methods=['GET', 'POST'])  # This is the transcription route
def transcribe_stream(): # This function will be called when the URL is of the form '/transcribe'
    client = speech.SpeechClient() # Create a speech client object
    config = speech.RecognitionConfig( # Create a recognition config object
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,
        model="telephony",
        speech_contexts=[speech.SpeechContext(phrases=["specific term1", "specific term2"], boost=20.0)]
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True) # Create a streaming recognition config object
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator() # Create an audio generator object
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator if content)
        responses = client.streaming_recognize(streaming_config, requests)
        if listen_print_loop(responses, stream): # Call the listen_print_loop function with the responses and stream objects
            print("Transcription process completed and stopped.")
            return redirect(url_for('lookup_number'))
        else:
            print("Transcription did not meet stop criteria.")
    return jsonify({"status": "transcription ended"})



@app.route('/', methods=["GET", "POST"]) # This is the home page route
def lookup_number(): # This function will be called when the URL is of the form '/'
    form = LookUpForm(csrf_enabled=False) # Create an instance of the LookUpForm class
    if form.validate_on_submit():
        number = form.number.data # Get the phone number from the form
        phone_number = phone.query.filter_by(number=number).first() # Query the database for the phone number
        if phone_number:
            return redirect(url_for('patient_info', number=number)) # Redirect to the patient_info route with the phone number
    return render_template('lookup.html', form=form)



@app.route('/<number>') # This is a dynamic route that accepts a phone number
def patient_info(number): # This function will be called when the URL is of the form '/<number>'
    formatted_number = number.replace('-', '').replace(' ', '') # Remove any hyphens or spaces from the phone number
    phone_number = phone.query.filter_by(number=formatted_number).first_or_404()
    return render_template('patient_info.html', phone_number=phone_number) # Render the patient_info.html template with the phone_number variable




if __name__ == '__main__': # This is the entry point of our application
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, ssl_context=('server.crt', 'server.key'), threaded=True)