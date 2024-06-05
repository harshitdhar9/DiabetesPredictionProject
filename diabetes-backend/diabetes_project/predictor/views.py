# predictor/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Define the path to the CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'predictor', 'data', 'Diabetes_prediction.csv')

# Load dataset
df = pd.read_csv(CSV_PATH)

# Prepare the data
y = df['Diagnosis']
X = df.drop(columns='Diagnosis')

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize the data
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# Train the model
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

def calculate_dpf(relatives):
    total_contribution = 0
    for relative in relatives:
        degree, age = relative
        if degree == 'first':
            weight = 0.5
        elif degree == 'second':
            weight = 0.25
        else:
            weight = 0
        total_contribution += weight / age
    return total_contribution

def calculate_bp_points(systolic, diastolic):
    systolic_weight = 0.6
    diastolic_weight = 0.4
    return (systolic * systolic_weight) + (diastolic * diastolic_weight)

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        pregnancies = int(data['pregnancies'])
        glucose = float(data['glucose'])
        systolic = int(data['systolic'])
        diastolic = int(data['diastolic'])
        skinthickness = float(data['skinthickness'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        age = float(data['age'])

        bloodpressure = calculate_bp_points(systolic, diastolic)
        relatives = [(rel['relation'], int(rel['age1'])) for rel in data['relatives']]
        dpf = round(calculate_dpf(relatives), 5)

        input_data = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]], 
                                  columns=X.columns)
        prediction = classifier.predict(input_data)

        result = "You may have diabetes." if prediction[0] == 1 else "You may not have diabetes."
        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request method.'}, status=400)
