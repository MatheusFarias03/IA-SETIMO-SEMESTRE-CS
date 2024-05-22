import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

def knn(n_neighbors, df, df_test):
    # Preparar os dados para treinamento.
    X_train = df.drop(columns=['target'])
    y_train = df['target']

    # Declarar o modelo.
    n_vizinhos = n_neighbors
    clf = neighbors.KNeighborsClassifier(n_neighbors)

    # Treinamento.
    clf.fit(X_train, y_train)

    # Predição.
    X_test = df_test.drop(columns=['target'])
    y_pred = clf.predict(X_test)

    df_test['target'] = y_pred
    print(df_test, '\n')


def exemplo_1():
    idade_meses = [6,7,7,5,8,4,4,3,1,1]
    target = ['A','B','B','A','A','A','C','C','B','A']
    data = {'idade_meses':idade_meses, 'target':target}
    df = pd.DataFrame(data)

    data_test = {'idade_meses':[7.2], 'target':['?']}
    df_test = pd.DataFrame(data_test)
    
    knn(3, df, df_test)
    knn(5, df, df_test)


def exemplo_2():
    loan = [1000, 2000, 2000, 7_000_000, 4000, 12000, 12000, 9_000_000, 1000, 3000]
    age = [22, 23, 34, 78, 18, 33, 35, 62, 27, 43]
    target = [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
    data = {'loan':loan, 'age':age, 'target':target}
    df = pd.DataFrame(data)

    data_test = {'loan':[6_050_000], 'age':[55], 'target':['?']}
    df_test = pd.DataFrame(data_test)
    knn(4, df, df_test)


if __name__ == '__main__':
    exemplo_2()
    
    