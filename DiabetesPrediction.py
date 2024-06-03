#Diabetes Prediction using logistsic regression method 
#Dataset from kaggle "Diabetes_prediction.csv"

"""
Diabetes pedigree function (DPF) is a function that calculates the likelihood of diabetes based on
a person's age and family history of diabetes. The function produces a score that ranges from 0.08 to 2.42,
with 0 representing a healthy person and 1 representing someone with diabetes. DPF is a positively skewed
variable with no zero values, and diabetics tend to have higher DPF scores than non-diabetics
"""

#The parameters used for predicting diabetes in a pregnant woman include pregnancies,bmi,glucose,bp,insulin,skin thickness,age

#importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Diabetes_prediction.csv')
result_df = df['Diagnosis']
features = df.drop(columns=['Diagnosis'])

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(features, result_df)

# Function to calculate diabetes pedigree function
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

def probcalc(model,X):
    log_odds=model.coef_*X+model.intercept_
    odds=np.exp(log_odds)
    probability=odds/(1+odds)
    return probability

# Predicting based on user inputs
pregnancy = int(input("Enter number of pregnancies: "))
glucose = float(input("Enter Glucose Level: "))
bloodpressure = float(input("Enter BloodPressure: "))
skinthickness = float(input("Enter SkinThickness: "))
insulin = float(input("Enter Insulin Level: "))
bmi = float(input("Enter BMI Level: "))
age = float(input("Enter Age: "))
n = int(input("Enter number of people in your family who have diabetes: "))
print()
list_1 = []
for i in range(n):
    relation = str(input("Enter relation (first/second): "))
    age1 = int(input("Enter age of relation: "))
    list_1.append((relation, age1))
    print()

calc_dpf = calculate_dpf(list_1)
dpf = round(calc_dpf, 5)

# Create a DataFrame for the input data
input_list=[pregnancy,glucose,bloodpressure,skinthickness,insulin,bmi,dpf,age]

# Predicting based on all features
predicted = model.predict(np.array([input_list]).reshape(1,-1))

print("Diabetes Pedigree Function is: ", dpf)
print("Input features: ", input_list)

probability = (probcalc(model,input_list))[0][1]
print("Probability of diabetes is: ", round(probability, 3))
print()
if predicted[0] == 1:
    print("You may have diabetes.")
else:
    print("You may not have diabetes.")
print()
print("Thank You!!")

