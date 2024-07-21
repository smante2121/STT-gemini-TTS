# Description: This file contains the functions that are used to process the audio stream from the microphone.
from google.cloud import speech
from google.cloud import texttospeech
import sounddevice as sd
from app import app, RATE
import numpy as np
import logging
from database import save_to_database
from app import chat
from extraction import extract_callback_number, extract_is_patient, extract_date_of_birth, extract_last_name_letters, extract_gender, extract_state, extract_symptom


def listen_print_loop(responses, stream): # Print the transcriptions and send the responses to the chatbot
    num_chars_printed = 0
    buffer = ""
    full_transcript = ""
    for response in responses: # Iterate through responses
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript # Get the transcript
        if transcript:
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))
            print(transcript + overwrite_chars, end='\r')
            num_chars_printed = len(transcript)
            if result.is_final: # If the result is final
                full_transcript += transcript + " "
                buffer += transcript
                print('\n' + buffer)

                if buffer.strip(): # If the buffer is not empty
                    response = chat.send_message(buffer) # Send the buffer to the chatbot
                    print(response.text) # Print the response from the chatbot
                    synthesize_text(response.text) # Synthesize the text
                buffer = ""
                num_chars_printed = 0

    return full_transcript



def synthesize_text(text): # Synthesize the text to speech
    client = texttospeech.TextToSpeechClient() # Create a text to speech client
    input_text = texttospeech.SynthesisInput(text=text) # Create a synthesis input
    voice = texttospeech.VoiceSelectionParams( # Create a voice selection parameter
        language_code="en-US",
        name="en-US-Standard-I",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig( # Create an audio config
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )
    response = client.synthesize_speech( # Synthesize the speech
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    audio_data = np.frombuffer(response.audio_content, dtype=np.int16) # Get the audio data
    sd.play(audio_data, samplerate=24000)
    sd.wait()



def process_stream(): # Process the audio stream
    global global_stream
    with app.app_context():  # Ensures the use of Flask's application context
        client = speech.SpeechClient()
        config = speech.RecognitionConfig( # Create a recognition config
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
            model="telephony"
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
        if global_stream is None: # If the stream is not active
            logging.error("Stream is not active.")
            return None
        with global_stream as stream: # Use the stream
            audio_generator = stream.generator() # Generate audio
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator if content and not content.isspace())
            responses = client.streaming_recognize(streaming_config, requests) # Get the responses
            try: # Try to process the stream
                full_transcript = listen_print_loop(responses, stream)
                if full_transcript:
                    details = process_full_transcript(full_transcript)
                    if details: # If the details are valid
                        save_to_database(details) # Save the details to the database
                        print("Transcription stopped, data processed and saved.")
                    else:
                        print("No valid data extracted.")
                else:
                    print("No transcript was processed.")
            except Exception as e:
                logging.error(f"Failed to process stream: {e}")


def process_full_transcript(full_transcript): # Process the full transcript
    details = { # Create a dictionary of details
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
