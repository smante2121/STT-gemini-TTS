# STT-Gemini-TTS Web Application

## Overview

Welcome to the STT-Gemini-TTS web application repository. This project is a research initiative to evaluate the accuracy of Google's Speech-to-Text (STT) API and the effectiveness of the Gemini conversational AI in managing conversation flow. The application leverages the Flask framework and incorporates HTML, CSS, and JavaScript for page design and functionality. 

## Features

- **Speech-to-Text Integration:** Utilizes Google's STT API to transcribe audio input captured via the computer microphone.
- **Conversational AI:** Employs Google's Gemini API to handle conversation flow and question management.
- **Text-to-Speech Output:** Uses Google's Text-to-Speech (TTS) API to vocalize responses generated by the Gemini AI.
- **Data Storage:** Extracted and validated data is stored in a database, with retrieval functionality based on user-provided phone numbers.
- **Web Interface:** Interactive web page with a start recording button to initiate the main function.

## Project Structure

- **app.py:** Main file that runs the application, manages conversation flow, phone look-up, and database operations.
- **stream.py:** Handles audio processing, conversation loop, and text synthesis.
- **database.py:** Manages database setup and related methods using SQLAlchemy.
- **microphone_stream.py:** Manages the microphone object and its functions.
- **extraction.py:** Contains methods for extracting data from transcriptions.
- **phone_class.py:** Defines the database schema using SQLAlchemy.


## Research and Findings

This project aimed to assess the suitability of Google's STT and Gemini APIs for a patient call system. The key findings are:

- **Accuracy of STT:** Google's STT was tested for transcription accuracy. It was found that alternative STT solutions might offer better performance.
- **Gemini Conversational Handling:** The Gemini AI sometimes deviated from its instructions, skipping questions or going off-topic. A hard-coded conversation handling approach may be more reliable.
- **Application in Patient Call Systems:** Insights gained will inform the development of a patient call system, where initial patient interactions are handled by AI before transferring to a healthcare professional. This system aims to improve efficiency and care quality by providing relevant information upfront.

- ## Future Work

- **Refine Conversation Handling:** Implement a more robust conversation flow management system.
- **Explore Alternative STT Solutions:** Test other speech-to-text APIs for better transcription accuracy.
- **Enhance Database Functionality:** Improve data storage and retrieval mechanisms.
