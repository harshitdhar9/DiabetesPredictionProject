#Diabetes Prediction using logistsic regression method 
#Dataset from kaggle "Diabetes_prediction.csv"

"""
Diabetes pedigree function (DPF) is a function that calculates the likelihood of diabetes based on
a person's age and family history of diabetes. The function produces a score that ranges from 0.08 to 2.42,
with 0 representing a healthy person and 1 representing someone with diabetes. DPF is a positively skewed
variable with no zero values, and diabetics tend to have higher DPF scores than non-diabetics
"""

#The parameters used for predicting diabetes in a pregnant woman include pregnancies,bmi,glucose,bp,insulin,skin thickness,age

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
   
df= pd.read_csv('Diabetes_prediction.csv')  
   
y= df.Diagnosis
Y = df['Diagnosis']
x= df.drop(columns='Diagnosis')
   
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  
   
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)

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
    
    bp_points = (systolic * systolic_weight) + (diastolic * diastolic_weight)
    return bp_points

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

pregnancy = int(input("Enter number of pregnancies: "))
glucose = float(input("Enter Glucose Level: "))
systolic = int(input("Enter systolic BloodPressure: "))
diastolic= int(input("Enter diastolic BloodPressure: "))
skinthickness = float(input("Enter SkinThickness(20-23): "))
insulin = float(input("Enter Insulin Level(60-100 after eating & 140 before eating for a normal human): "))
bmi = float(input("Enter BMI Level: "))
age = float(input("Enter Age: "))

bloodpressure=calculate_bp_points(systolic, diastolic)

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

input_data = pd.DataFrame([[pregnancy,glucose,bloodpressure,skinthickness,insulin,bmi,dpf,age]], columns=features)
prediction = dtree.predict(input_data)

if prediction[0] == 1:
    print("You may have diabetes.")
else:
    print("You may not have diabetes.")

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

cm = confusion_matrix(y_test, y_pred)

#Below confusion matrix evaluates the model
"""
plt.figure(figsize=(10, 7))
plt.title('Confusion Matrix')
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()

plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
plt.yticks([0, 1], ['No Diabetes', 'Diabetes'])

plt.text(0, 0, cm[0, 0], ha='center', va='center', color='white')
plt.text(0, 1, cm[0, 1], ha='center', va='center', color='black')
plt.text(1, 0, cm[1, 0], ha='center', va='center', color='black')
plt.text(1, 1, cm[1, 1], ha='center', va='center', color='white')

plt.show()
"""