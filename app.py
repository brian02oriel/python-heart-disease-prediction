from flask import Flask, request, jsonify
import joblib
import numpy as np
from classes.ModelPipeline import ModelPipeline

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        pipeline = ModelPipeline(data)
        data = pipeline.execute()
        model = joblib.load('model-registry/heart_disease_prediction.joblib')
        prediction = model.predict_proba(data)
        print(prediction)
    except Exception as error:
        print(error)
        return jsonify({'error': error})
    else:
        return jsonify(prediction.tolist())

@app.route('/', methods=['GET'])
def test():
    return 'Heart Disease Prediction Online!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
