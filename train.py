import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import os

#Read the voice dataset
raw_data = pd.read_csv("voice/voice.csv")

#Select relevant data and transform labels
relevant_data = raw_data[['meanfreq', 'sd', 'IQR', 'label']]
labels_as_numbers = relevant_data.replace({'label':{'male':0,'female':1}})
data = labels_as_numbers

#Prepare data for modeling
data_train, data_test = train_test_split(data, random_state=0, test_size=.2)
n_params = data_train.shape[1]-1

#Scale data
scaler = StandardScaler()
scaler.fit(data_train.iloc[:,0:n_params])

X_train = scaler.transform(data_train.iloc[:,0:n_params])
X_test = scaler.transform(data_test.iloc[:,0:n_params])
y_train = list(data_train['label'].values)
y_test = list(data_test['label'].values)

print("Training models")

#Train decision tree model
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Decision Tree")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

#Train random forest model
forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
print("Random Forests")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

#Train gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("Gradient Boosting")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

#Train support vector machine model
svm = SVC().fit(X_train, y_train)
print("Support Vector Machine")
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test, y_test)))

#Train neural network model
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("Multilayer Perceptron")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

models_dir = 'models'

models = {'tree': tree, 'forest': forest, 'gradient': gbrt, 'svm': svm, 'mlp': mlp}

try:
    os.mkdir(models_dir)
except OSError:
    print('Creation of %s directory failed', models_dir)
else:
    print("Dumping models")
    for model_name, model in models.items():
        filename = os.path.join(models_dir, model_name + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

print("Dumping scaler")
filename = os.path.join(models_dir, 'scaler.pkl')
with open(filename, 'wb') as file:
            pickle.dump(scaler, file)
