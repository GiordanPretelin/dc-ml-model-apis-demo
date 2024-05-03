from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))

            query_dict = query.to_dict(orient="records")
            query = pd.get_dummies(pd.DataFrame(query_dict))

            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = lr.predict(query)
            response = json.dumps({"prediction": prediction.tolist()}, cls=NumpyEncoder)

            return response
        except:
            return jsonify({"trace": traceback.format_exc()})
    else:
        print("Train the model first")
        return "No model here to use"


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
