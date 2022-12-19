
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from tensorflow import keras


app = Flask(__name__)
model = keras.models.load_model("models/fold9.P-1D-CNN.099-0.819.h5")
labels = ["A", "B", "C", "D", "E"]


def reshape_signal(signal):
    signal = np.expand_dims(signal, axis=1)
    signal = np.expand_dims(signal, axis=0)
    return np.asarray(signal)


@app.route("/", methods=['GET'])
def welcome_page():
    return jsonify({"message": "welcome to the Ep-Det-Ml webserver"})


@app.route("/predict", methods=['POST'])
def do_prediction():
    json = request.get_json()

    data = np.array(json["data"])
    reshaped_signal = reshape_signal(data)
    prediction = model.predict(reshaped_signal)[0]
    label_index = np.argmax(prediction)

    response = {
        "prediction": {
            "label": str(labels[label_index]),
            "confidence":  str(prediction[label_index])
        }
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2000)
