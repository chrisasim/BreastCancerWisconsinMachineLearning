import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('wdbc.data', sep=",", header=None)
data.columns=['ID', 'Diagnosis', 'radius1','texture1', 'perimeter1', 'area1', 'smoothness1', 'compactness1', 'concavity1', 'concave_points1', 'symmetry1','fractal_dimension1', 'radius1','texture2','perimeter2', 'area2','smoothness2', 'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3', 'compactness3', 'concavity3','concave_points3', 'symmetry3', 'fractal_dimension3']

#Παρακάτω ακολουθεί το πρώτο ερώτημα
data['Diagnosis'].replace(['M', 'B'], [1, 0], inplace=True)

#παρακάτω ακολουθεί το δεύτερο ερώτημα.
data['Diagnosis'].describe()
mean = data['Diagnosis'].describe().loc['mean']
std = data['Diagnosis'].describe().loc['std']
variance = std*std
print(data['Diagnosis'].describe())

#παρακάτω ακολουθεί το τρίτο ερώτημα.
dataMeanAndVariance = {'Mean': mean, 'Var': variance}
meanvar = list(dataMeanAndVariance.keys())
values = list(dataMeanAndVariance.values())
plt.bar(meanvar, values, color='maroon', width=0.4)
plt.xlabel("Mean and variance")
plt.ylabel("Values")
plt.show()

#παρακάτω ακολουθεί το τέταρτο ερώτημα.
y = data['Diagnosis']
X = data.loc[:, ~data.columns.isin(['ID', 'Diagnosis'])]
#Άρα έχουμε δύο μεταβλητές την Χ και την y.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)
print(model.score(X_test, y_test))
model.predict_proba(X_test)
print(model.predict_proba(X_test))


