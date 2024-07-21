# Description: This file contains functions to extract information from a buffer of text.
import re


def extract_callback_number(buffer): # Extract the callback number from the buffer
    match = re.search(r'\b(\d{3}[- ]?\d{3}[- ]?\d{4})\b', buffer)
    if match:
        return re.sub(r'[- ]', '', match.group(1))
    return None

def extract_is_patient(buffer): # Extract whether the caller is the patient from the buffer
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

def extract_date_of_birth(buffer): # Extract the date of birth from the buffer
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

def extract_last_name_letters(buffer): # Extract the first three letters of the last name from the buffer
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

def extract_state(buffer): # Extract the state name from the buffer
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

def extract_symptom(buffer): # Extract the main symptom from the buffer
    # Enhance symptom capture by ignoring the echo of the question
    pattern = r'(?<=tell me your main symptom or reason for the call today\.\s*)[A-Za-z].*?([^.]+)\.'
    match = re.search(pattern, buffer, re.IGNORECASE | re.DOTALL)
    if match:
        # Clean up response from any lead text echoed from the question
        response = match.group(1).strip()
        return response.replace('I\'m calling because ', '').replace('My symptom is ', '')
    return None