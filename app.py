from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = lr.predict(query)

            return jsonify({"prediction": list(prediction)})
        except:
            return jsonify({"trace": traceback.format_exc()})
    else:
        print("Train the model first")
        return "No model here to use"


@app.route("/", methods=["POST", "GET"])
def welcome():
    return "Welcome to the api!"


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    lr = joblib.load("model.pkl")
    print("Model loaded")
    model_columns = joblib.load("model_columns.pkl")
    print("Model columns loaded")
    app.run(port=port, debug=True)
