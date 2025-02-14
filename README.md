# Sentiment Analysis Using LSTM

## Overview
This project implements a sentiment analysis model using **Long Short-Term Memory (LSTM)** networks to classify movie reviews as **positive, negative, or neutral**. It utilizes **Gradio** for an interactive user interface and **Matplotlib** to visualize sentiment confidence.

## Features
- Uses an LSTM model trained on the **IMDB dataset**.
- Implements **Gradio** for a user-friendly interface.
- Generates a **pie chart visualization** of sentiment confidence.
- Supports text input for real-time sentiment classification.

## Implementation Details
- **Deep Learning Model:** LSTM-based sentiment classification.
- **Tokenizer:** Pre-trained tokenizer for text preprocessing.
- **User Interface:** Implemented using Gradio.
- **Visualization:** Confidence levels plotted using Matplotlib.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install tensorflow gradio matplotlib pickle5
```

## How to Run
1. Load the trained **LSTM model** and **tokenizer**:
```python
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import gradio as gr

model = tf.keras.models.load_model("my_model.keras")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

2. Define the sentiment classification function:
```python
def classify_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)
    
    positive_confidence = prediction[0][0]
    negative_confidence = 1 - positive_confidence
    
    if positive_confidence > 0.7:
        sentiment = "Positive"
    elif negative_confidence > 0.7:
        sentiment = "Negative"
    elif 0.4 < positive_confidence < 0.6 and 0.4 < negative_confidence < 0.6:
        sentiment = "Natural"
    else:
        sentiment = "Neutral"
    
    labels = ["Positive", "Negative"]
    confidences = [positive_confidence, negative_confidence]
    
    plt.figure(figsize=(6, 6))
    plt.pie(confidences, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Sentiment Confidence Distribution")
    plt.axis('equal')
    plt.savefig("sentiment_distribution.png")
    plt.close()
    
    return sentiment, f"{positive_confidence:.2f}", f"{negative_confidence:.2f}", "sentiment_distribution.png"
```

3. Create a **Gradio** interface:
```python
iface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(label="Enter a movie review"),
    outputs=[
        gr.Textbox(label="Sentiment Analysis Result"),
        gr.Textbox(label="Positive Confidence"),
        gr.Textbox(label="Negative Confidence"),
        gr.Image(label="Sentiment Confidence Distribution")
    ],
    title="Movie Review Sentiment Analysis",
    description="Enter a movie review to get a prediction with sentiment and confidence."
)

iface.launch(debug=True)
```

## Example Output
- **User Input:** "This movie was fantastic! The storyline was engaging."
- **Prediction:** Positive
- **Confidence Scores:**
  - Positive: 85%
  - Negative: 15%
- **Visualization:** Pie chart displaying confidence scores.

## Future Improvements
- Expand dataset for better accuracy.
- Implement **word embeddings (Word2Vec, GloVe)** for improved text representation.
- Enhance UI with additional feedback and examples.

## Acknowledgments
This project is inspired by **IMDB Sentiment Analysis** and utilizes **TensorFlow** and **Gradio** for deep learning and UI integration.

