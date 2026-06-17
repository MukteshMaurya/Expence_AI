import joblib

model = joblib.load("model.pkl")

def predict_category(text):

    # Convert to lowercase
    text = text.lower()

    # IMPORTANT KEYWORDS
    keywords = {
        "Food": [
            "zomato", "swiggy", "pizza",
            "burger", "food", "restaurant"
        ],

        "Transport": [
            "uber", "ola", "cab",
            "metro", "bus", "petrol"
        ],

        "Entertainment": [
            "netflix", "spotify",
            "movie", "gaming"
        ],

        "Shopping": [
            "amazon", "flipkart",
            "shopping", "clothes"
        ],

        "Bills": [
            "electricity", "wifi",
            "bill", "recharge",
            "gas", "water"
        ],

        "Education": [
            "college", "book",
            "course", "fees"
        ],

        "Healthcare": [
            "doctor", "medicine",
            "hospital", "medical"
        ]
    }

    # KEYWORD MATCHING
    for category, words in keywords.items():

        for word in words:

            if word in text:
                return category

    # ML PREDICTION
    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    max_prob = max(probabilities)

    # LOW CONFIDENCE
    if max_prob < 0.20:
        return "Unknown"

    return prediction