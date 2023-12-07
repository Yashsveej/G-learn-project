from flask import Flask, request, jsonify
import joblib
import numpy as np

app= Flask(__name__)

#load the saved model
model_filename = 'random_forest_model.joblib'
loaded_model = joblib.load(model_filename)

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #get input data from request
        data= request.get_json(force= True)
        features = np.array(data['features']).reshape(1,-1)
        #make predictions using the loaded model
        prediction = loaded_model.predict(features)
        #return the prediction as json
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__== '__main__':
    app.run(port=5000, debug=True)
