from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained models
cv = pickle.load(open(r"models/cv.pkl", "rb"))       
clf = pickle.load(open(r"models/clf.pkl", "rb"))     

@app.route("/", methods=["GET", "POST"])
def home():
    email = ""
    predictions = None

    if request.method == "POST":
        email = request.form.get("content", "")
        if email.strip():  # make sure email is not empty
            tokenized_email = cv.transform([email])
            predictions = clf.predict(tokenized_email)[0]
            # Normalize predictions to match your HTML logic
            predictions = 1 if predictions == 1 else -1

    return render_template("index.html", email=email, predictions=predictions)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
