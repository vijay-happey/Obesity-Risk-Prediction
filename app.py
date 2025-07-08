from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the label encoder
label_encoder_path = 'label_encoder.pkl'
if os.path.exists(label_encoder_path):
    label_encoder = joblib.load(label_encoder_path)
    print("Label encoder loaded successfully.")
else:
    print(f"Error: {label_encoder_path} not found. Label encoder not loaded.")
    label_encoder = None

# Load the saved model
model_path = 'trained_model.pkl'
if os.path.exists(model_path):
    best_model = joblib.load(model_path)
    print("Model loaded successfully.")
else:
    print(f"Error: {model_path} not found. Model not loaded.")
    best_model = None

@app.route('/')
def index():
    return render_template('predict_final.html')

@app.route('/predict', methods=['POST'])
def predict_obesity():
    if best_model is None or label_encoder is None:
        return jsonify({'error': 'Model or label encoder not loaded. Unable to make predictions.'}), 500

    data = request.json

    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'Gender': [data['gender']],
            'Age': [float(data['age'])],
            'Height': [float(data['height'])],
            'Weight': [float(data['weight'])],
            'family_history_with_overweight': [data['family_history']],
            'FAVC': [data['favc']],
            'FCVC': [float(data['fcvc'])],
            'NCP': [float(data['ncp'])],
            'CAEC': [data['caec']],
            'SMOKE': [data['smoke']],
            'CH2O': [float(data['ch2o'])],
            'SCC': [data['scc']],
            'FAF': [float(data['faf'])],
            'TUE': [float(data['tue'])],
            'CALC': [data['calc']],
            'MTRANS': [data['mtrans']]
        })

        # Make prediction using the loaded model
        prediction = best_model.predict(input_data)
        result = label_encoder.inverse_transform(prediction)[0]

        return jsonify({'prediction': result})
    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)