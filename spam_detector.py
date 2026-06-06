import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "message": [
        "Win a free prize now",
        "Congratulations you won a lottery",
        "Claim your reward today",
        "Urgent call this number",
        "Hello how are you",
        "Let’s meet tomorrow",
        "Are you coming to class",
        "Good morning friend",
        "Free money waiting for you",
        "You have been selected for offer"
    ],
    "label": [
        "spam",
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham",
        "ham",
        "spam",
        "spam"
    ]
}

df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
user_message = input("Enter a message: ")

test_vector = vectorizer.transform([user_message])

prediction = model.predict(test_vector)

print("Prediction:", prediction[0])
