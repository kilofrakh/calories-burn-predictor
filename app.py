import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.svm import SVC
from xgboost import XGBRegressor
from flask import Flask, render_template, request
import warnings
import json

data = {
    "name":{
        "inputs": [],
        "predictions": []
    }
}
def save_data():
    with open('data.json', 'w') as f:
        json.dump(data, f)

app = Flask(__name__)


warnings.filterwarnings('ignore')

df = pd.read_csv('calories_burned_data.csv')

df.replace({'Male': 0, 'Female': 1}, inplace=True)

features = ['Running Time(min)', 'Gender', 'Age', 'Weight(kg)', 'Height(cm)', 'Running Speed(km/h)', 'Distance(km)']

df = df[[col for col in features if col in df.columns] + ['Calories Burned']]

x = df.drop(columns=['Calories Burned'])
target = df['Calories Burned']

X_train, X_val, Y_train, Y_val = train_test_split(x, target, test_size=0.1, random_state=22)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

r = XGBRegressor()
r.fit(X_train, Y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        try:
            user_input = [
            str(request.form['Name']),
            float(request.form['Duration']),
            float(request.form['Gender']),
            float(request.form['Age']),
            float(request.form['Weight']),
            float(request.form['Height']),
            float(request.form['Running Speed(km/h)']),
            float(request.form['Distance(km)'])
            ]
            data['name']['inputs'].append(user_input)
            user_input = user_input.drop('Name')
        except:
            return render_template('index.html', prediction='Please enter valid input')
        user_input = np.array(user_input).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)
        prediction = r.predict(user_input_scaled)
        data['name']['predictions'].append(prediction[0])
        save_data()
        return render_template('index.html', prediction=f'Predicted Calories: {prediction[0]:.2f}')


    


if __name__ == '__main__':
    app.run(debug=True)
