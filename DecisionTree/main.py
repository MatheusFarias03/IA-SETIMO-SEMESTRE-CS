import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



if __name__ == '__main__':

    # Importing data.
    data = load_breast_cancer()
    dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

    # Divide the data into training and testing.
    X = dataset.copy()
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Train the decision tree.
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Return predictions.
    predictions = clf.predict(X_test)
    print(f'predictions: {predictions} \n')

    # Confusion matrix.
    conf_matrix = confusion_matrix(y_test, predictions, labels=[0,1])

    # Check accuracy score (correct / total predictions).
    acc_sc = accuracy_score(y_test, predictions)
    print(f'accuracy_score: {acc_sc} \n')

    # Check the precision.
    prec = precision_score(y_test, predictions)
    print(f'precision_score: {prec} \n')