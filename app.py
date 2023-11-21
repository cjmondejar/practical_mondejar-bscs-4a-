import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the SVM model trained for credit card eligibility
model = pickle.load(open('PRACTICAL.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # feature = request.form.values('spending')
    user_input = [float(request.form[field]) for field in
                  ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                   'diabetes']]
    user_input = np.array(user_input).reshape(1, -1)
    prediction = model.predict(user_input)
    print(prediction)
    result = 'Diabetic' if prediction == 0 else 'Not Diabetic'

    return render_template('index.html', prediction_output=f'Person is {result}')


if __name__ == "__main__":
    app.run(debug=True)
