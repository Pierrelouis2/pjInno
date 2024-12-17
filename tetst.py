from emotionclassifier import EmotionClassifier


def EmotionClassifierPerso(text) :
    # Initialize the classifier with the default model
    classifier = EmotionClassifier()

    # Classify a single text
    text = "I am very happy today!"
    result = classifier.predict(text)
    print("Emotion:", result['label'])
    print("Confidence:", result['confidence'])
    return result['label']