from flask import Flask, request, jsonify
import joblib
import numpy as np
from classes.ModelPipeline import ModelPipeline

app = Flask(__name__)

# Load the model
model = joblib.load('model-registry/heart_disease_prediction.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        pipeline = ModelPipeline(data)
        data = pipeline.execute()
        #prediction = model.predict(np.array(data['input']).reshape(1, -1))
    except Exception as error:
        print(error)
        return jsonify({'error': error})
    else:
        return data.to_json() #jsonify({'prediction': prediction.tolist()})

@app.route('/', methods=['GET'])
def test():
    return 'Heart Disease Prediction Online!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
