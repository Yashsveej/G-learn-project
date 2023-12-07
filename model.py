#import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#load iris data
iris = load_iris()
X= iris.data
y= iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 42)

#initialize the random forest classifier 
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

#train random forest model
random_forest.fit(X_train, y_train)

#make predictions on test set 
y_pred = random_forest.predict(X_test)

#evaluate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f} ")

#save the trained model to a file 
model_filename = 'random_forest_model.joblib'
joblib.dump(random_forest, model_filename)
print(f"Model saved to {model_filename}")
