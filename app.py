import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
# from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
import random
import streamlit as st

st.title('Orchid : A mental health chatbot')

# model = keras.models.load_model('orchid.h5')

data=pd.read_csv('orchid - 20200325_counsel_chat.csv.csv')

le = LabelEncoder()
le.fit(data['questionTitle'])
data['label'] = le.transform(data['questionTitle'])

# Tokenize the input data
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data['questionText'])

def generate_response(input_text):
    # Find the question in the dataframe that is closest to the input text
    closest_question = process.extractOne(input_text, data['questionText'])[0]
    # Get the corresponding label for the closest question
    label = data.loc[data['questionText'] == closest_question]['label'].values[0]
    # Get a random answer belonging to the corresponding questionTitle
    possible_answers = data.loc[data['label'] == label]['answerText'].values
    response = random.choice(possible_answers)
    return response

# Start the chatbot
st.write("Welcome to Orchid!")
st.write("Type 'exit' to end the chatbot.")
def get_text():

    input_text = st.text_input(f"You : ","hi")
    return input_text

while True:
    user_input = get_text()
    if user_input in ('hi', 'yo', 'hey', 'hello', 'greetings', 'help'):
        st.write(
            "Hello I'm Orchid and I will answer all your mental health related quries. Just ask the question straightaway")
    elif user_input == 'exit':
        st.write("Chatbot: Goodbye!")
        break
    else:
        response = generate_response(user_input)
        st.write("Chatbot: ", response)
