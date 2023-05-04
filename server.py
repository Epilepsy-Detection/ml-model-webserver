
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, Response
from tensorflow import keras


app = Flask(__name__)

#TODO: Manage the security of the app to be only allowed from our backend

# TODO: Migrate to fetch latest model from s3 model, and have a default local, s3 model.
# TODO: Add way or facility to choose the model (in the route)
model = keras.models.load_model("models/fold9.P-1D-CNN.099-0.819.h5")

# Check how can you make this generalized for 3 labels and 5 labels (handle multi-models)
labels = ["A", "B", "C", "D", "E"]

# TODO: Agree with ML Team if this is global and not specific
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

    if data.size != 512:
        return jsonify(error=404, text=str("Sample size must be 512")), 404
    

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
