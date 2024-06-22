from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
from classes.ModelPipeline import ModelPipeline

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    try:
        data = request.get_json(force=True)
        pipeline = ModelPipeline(data)
        data = pipeline.execute()
        model = joblib.load('model-registry/heart_disease_prediction.joblib')
        if model:
            print("Model loaded successfully.")
        prediction = model.predict_proba(data)
    except Exception as error:
        print(error)
        return jsonify({'error': error})
    else:
        proba_array = prediction.tolist()[0]
        print(proba_array)
        return jsonify({
                'positive': proba_array[0] * 100,
                'negative': proba_array[1] * 100
            })

@app.route('/', methods=['GET'])
def test():
    return 'Heart Disease Prediction Online!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
