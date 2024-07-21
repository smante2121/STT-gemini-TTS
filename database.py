# Description: This file contains the database model and the function to save the details to the database.
import logging
from flask_sqlalchemy import SQLAlchemy
from app import app

db = SQLAlchemy(app)

class phone(db.Model): # Create a phone class
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(15), index=True, unique=True)
    patient = db.Column(db.String(100), index=True, unique=False)
    dob = db.Column(db.String(100), index=True, unique=False)
    lastName = db.Column(db.String(100), index=True, unique=False)
    gender = db.Column(db.String(100), index=True, unique=False)
    state = db.Column(db.String(100), index=True, unique=False)
    symptom = db.Column(db.String(100), index=True, unique=False)

    def __repr__(self): # Return the phone number, patient name, date of birth, last name, state, and symptom
        return f"Phone number: {self.number} Name: {self.patient} DOB: {self.dob} Last Name: {self.lastName} State: {self.state} Symptom: {self.symptom}"



def save_to_database(details): # Save the details to the database
    try:
        new_entry = phone(**details)
        db.session.add(new_entry)
        db.session.commit()
        logging.info(f"Saved to database: {new_entry}")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error saving to database: {e}")

