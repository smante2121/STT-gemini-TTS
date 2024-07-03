import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap
from IPython.display import Markdown

load_dotenv()
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


response = model.generate_content("What is the meaning of life?")

print(response.text)



def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

