from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the data
df = pd.read_csv('Plant_Parameters.csv')

# Assuming your columns are named 'Temperature', 'pH', 'Soil_Moisture', and so on
feature_columns = ['Temperature', 'pH', 'Soil_Moisture', 'Column4', 'Column5', 'Column6', 'Column7', 'Column8', 'Column9']
target_column = 'Result'

# Split the data into features (x) and target variable (y)
x = df[feature_columns].values
y = df[target_column].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form and convert to float
            temperature = float(request.form['temperature'])
            pH = float(request.form['pH'])
            soil_moisture = float(request.form['soil_moisture'])

            # Make prediction
            input_values = [[temperature, pH, soil_moisture, None, None, None, None, None, None]]
            input_values = pd.DataFrame(input_values, columns=feature_columns)

            for col in input_values.columns:
                if input_values[col].isnull().any():
                    input_values[col].fillna(X_train[:, feature_columns.index(col)].mean(), inplace=True)

            # Standardize the input using the same scaler used for training
            input_values_scaled = sc.transform(input_values)

            # Make the prediction
            prediction = classifier.predict(input_values_scaled)[0]

            # Redirect to the result page with the prediction
            return redirect(url_for('result', prediction=prediction))

        except ValueError:
            # Handle the case where the input values cannot be converted to float
            return render_template('index.html', prediction="Invalid input. Please enter valid float values.")

    return render_template('index.html')

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
