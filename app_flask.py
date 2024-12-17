from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from emotionclassifier import EmotionClassifier


app = Flask(__name__)

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

def compare_predictions(pred1, conf1, pred2, conf2):
    if pred1 == pred2:
        return pred1, max(conf1, conf2)
    else:
        return (pred1, conf1) if conf1 > conf2 else (pred2, conf2)

@app.route("/", methods=["GET", "POST"])
def index():
    person1_indicator = 0
    person2_indicator = 0
    global_indicator = 0
    chat_history = []
    final_label = ""
    final_confidence = 0.0
    messages = []
    emotions = []

    if request.method == "POST":
        input_text = request.form["conversation"]
        if input_text:
            messages_input = input_text.split('-')

            for i, message in enumerate(messages_input):
                if message.strip():
                    input_sequence = tokenizer.texts_to_sequences([message.strip()])
                    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
                    
                    # Model 1 prediction
                    prediction1 = loaded_model.predict(padded_input_sequence)
                    predicted_label1 = label_encoder.inverse_transform([np.argmax(prediction1[0])])[0]
                    confidence1 = np.max(prediction1[0])
                    
                    # Model 2 prediction
                    result2 = classifier.predict(message.strip())
                    predicted_label2 = result2['label']
                    confidence2 = result2['confidence']
                    
                    # Compare predictions
                    final_label, final_confidence = compare_predictions(predicted_label1, confidence1, predicted_label2, confidence2)
                    
                    # Adjust the indicator based on the final emotion
                    impact = adjust_indicator(final_label)
                    global_indicator += impact
                    if i % 2 == 0:  # Assuming person 1 starts the conversation
                        person1_indicator += impact
                        chat_history.append(f'<div class="person1">Person 1: {message.strip()} ({final_label})</div>')
                    else:
                        person2_indicator += impact
                        chat_history.append(f'<div class="person2">Person 2: {message.strip()} ({final_label})</div>')

                    # Print the best emotion found
                    print(f"Best Emotion: {final_label}, Confidence: {final_confidence}")

                    messages.append(message.strip())
                    emotions.append(final_label)

            # Normalize the global indicator to a range of 0 to 100 for the progress bar
            normalized_global_indicator = (global_indicator + 10) * 5  # Assuming the range is -10 to 10

            return render_template("index.html", messages=messages, emotions=emotions, chat_history=chat_history, person1_indicator=person1_indicator, person2_indicator=person2_indicator, normalized_global_indicator=normalized_global_indicator, final_label=final_label, final_confidence=final_confidence)

    return render_template("index.html", messages=messages, emotions=emotions, chat_history=chat_history, person1_indicator=person1_indicator, person2_indicator=person2_indicator, normalized_global_indicator=0, final_label=final_label, final_confidence=final_confidence)

if __name__ == "__main__":
    app.run(debug=True)
