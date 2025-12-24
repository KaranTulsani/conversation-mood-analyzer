# Map DailyDialog emotions → sentiment
EMOTION_TO_SENTIMENT = {
    0: 0,  # neutral → neutral
    1: 1,  # anger → negative
    2: 1,  # disgust → negative
    3: 1,  # fear → negative
    4: 2,  # happiness → positive
    5: 1,  # sadness → negative
    6: 2   # surprise → positive
}

SENTIMENT_LABELS = {
    0: "neutral",
    1: "negative",
    2: "positive"
}
