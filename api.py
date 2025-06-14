from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('output/house_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "🏡 Welcome to the House Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        area = data['area']
        bedrooms = data['bedrooms']
        bathrooms = data['bathrooms']
        parking = data['parking']

        input_data = np.array([[area, bedrooms, bathrooms, parking]])
        prediction = model.predict(input_data)[0]

        return jsonify({
            'predicted_price': int(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
