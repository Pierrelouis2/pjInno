import streamlit as st
import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from .tetst import EmotionClassifierPerso


# Load and preprocess the data
data = pd.read_csv("train.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Load the saved model architecture
json_file = open("model_architecture.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the saved model weights
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.h5")


classifier = EmotionClassifier()
def adjust_indicator(emotion):
    # Define how each emotion affects the indicator
    emotion_impact = {
        "joy": 2,
        "happy": 1,
        "sadness": -1,
        "angry": -2,
        "neutral": 0,
        # Add other emotions and their impacts as needed
    }
    return emotion_impact.get(emotion, 0)

# Streamlit app
def main():
    st.title("Text Emotion Classification")
    person1_indicator = 0
    person2_indicator = 0
    global_indicator = 0
    # User input
    input_text = st.text_area("Enter the conversation (use '-' before each new message):", "")

    if st.button("Analyze Conversation"):
        print("--------------------USING KERAS MODEL PREDICTION-------------------")
        if input_text:
            messages = input_text.split('-')
            
            chat_history = []

            for i, message in enumerate(messages):
                if message.strip():
                    input_sequence = tokenizer.texts_to_sequences([message.strip()])
                    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
                    prediction = loaded_model.predict(padded_input_sequence)
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                    print(predicted_label)
                    print("--------------------USING PIP MODULE PREDICTION-------------------")
                    # Classify a single text
                    result = classifier.predict(message.strip())
                    print("Emotion:", result['label'])
                    print("Confidence:", result['confidence'])
                    # Adjust the indicator based on the emotion
                    impact = adjust_indicator(predicted_label)
                    global_indicator += impact
                    if i % 2 == 0:  # Assuming person 1 starts the conversation
                        person1_indicator += impact
                        chat_history.append(f'<div class="person1">Person 1: {message.strip()} ({predicted_label})</div>')
                    else:
                        person2_indicator += impact
                        chat_history.append(f'<div class="person2">Person 2: {message.strip()} ({predicted_label})</div>')

            # Display chat history
            st.markdown(
                """
                <style>
                .chat-container {
                    max-height: 300px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                }
                .person1 {
                    text-align: left;
                    background-color: #d1e7dd;
                    padding: 5px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }
                .person2 {
                    text-align: right;
                    background-color: #f8d7da;
                    padding: 5px;
                    border-radius: 5px;
                    margin-bottom: 5px;
                }
                </style>
                <div class="chat-container">
                """ + "".join(chat_history) + """
                </div>
                """, unsafe_allow_html=True
            )

            # Normalize the global indicator to a range of 0 to 100 for the progress bar
            normalized_global_indicator = (global_indicator + 10) * 5  # Assuming the range is -10 to 10

            st.write(f"Person 1 Indicator: {person1_indicator}")
            st.write(f"Person 2 Indicator: {person2_indicator}")
            st.progress(normalized_global_indicator, "<-- Bad Conversation | Good Conversation -->")

if __name__ == "__main__":
    main()
