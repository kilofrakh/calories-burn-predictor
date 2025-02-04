import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape
from sklearn.svm import SVC
from xgboost import XGBRegressor
from flask import Flask, render_template, request
import warnings
import os


data = {
    "name": {
        "inputs": [],
        "predictions": []
    }
}


def save_to_csv():
    file_path = 'data.csv'
    fieldnames = ['Name', 'Duration', 'Gender', 'Age', 'Weight', 'Height', 'Running Speed(km/h)', 'Distance(km)', 'Prediction']
    
    new_data = []
    for input_data, prediction in zip(data['name']['inputs'], data['name']['predictions']):
        new_data.append({
            'Name': input_data['Name'],
            'Duration': input_data['Duration'],
            'Gender': input_data['Gender'],
            'Age': input_data['Age'],
            'Weight': input_data['Weight'],
            'Height': input_data['Height'],
            'Running Speed(km/h)': input_data['Running Speed(km/h)'],
            'Distance(km)': input_data['Distance(km)'],
            'Prediction': prediction['Prediction']
        })
    
    new_df = pd.DataFrame(new_data, columns=fieldnames)
    
   
    if os.path.isfile(file_path):

        existing_df = pd.read_csv(file_path)

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        
        combined_df = new_df
    
    
    combined_df.to_csv(file_path, index=False)
    
    
    data['name']['inputs'].clear()
    data['name']['predictions'].clear()

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
        user_name = request.form['Name']
        user_input = [
            float(request.form['Duration']),
            float(request.form['Gender']),
            float(request.form['Age']),
            float(request.form['Weight']),
            float(request.form['Height']),
            float(request.form['Running Speed(km/h)']),
            float(request.form['Distance(km)'])
        ]
        data['name']['inputs'].append({
            "Name": user_name,
            "Duration": user_input[0],
            "Gender": user_input[1],
            "Age": user_input[2],
            "Weight": user_input[3],
            "Height": user_input[4],
            "Running Speed(km/h)": user_input[5],
            "Distance(km)": user_input[6]
        })
    except KeyError as e:
        return render_template('index.html', prediction=f'Missing input: {e.args[0]}')
    except ValueError:
        return render_template('index.html', prediction='Please enter valid input')
    
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    prediction = r.predict(user_input_scaled)
    data['name']['predictions'].append({
        "Name": user_name,
        "Prediction": prediction[0]
    })
    
    save_to_csv()
    
    return render_template('index.html', prediction=f'Calories burnt: {prediction[0]}')



if __name__ == '__main__':
    app.run(debug=True)

