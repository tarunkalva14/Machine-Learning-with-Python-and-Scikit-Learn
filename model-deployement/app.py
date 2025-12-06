from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load CORRECT models
cv = pickle.load(open(r"models/cv.pkl", "rb"))     # CountVectorizer
clf = pickle.load(open(r"models/clf.pkl", "rb"))   # Classifier (MultinomialNB)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get("content")

    # Transform using vectorizer
    tokenized_email = cv.transform([email])

    # Predict using model
    predictions = clf.predict(tokenized_email)
    prediction = predictions[0]

    # Normalize to 1/-1 if needed
    prediction = 1 if prediction == 1 else -1

    return render_template("index.html", prediction=prediction, email=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
