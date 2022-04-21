import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model          #Importamos Linear_model para la regresion lineal


def Run():
    data = pd.read_csv(
        r"C:\Users\user\Desktop\Curso de Ciencia de Datos\Data Science IA Fundamentals\Codigos_Pycharm\data.csv")  # Cargamos el dataset en formato csv

    # plt.scatter(data.Exposure,data.PEFR)    #Diagrama de dispersion

    model = linear_model.LinearRegression()  # Cargamos los metodos de la funcion en la variable

    model.fit(data[['Exposure']], data['PEFR'])  # Con el fit hacemos la regresion lineas

    intercept = model.intercept_  # b0, cruce con el eje de las ordenadas
    coef = model.coef_[0]  # Pendiente de la recta

    X = [i for i in range(26)]  # Creamos una lista del 0 al 25
    X = np.array(X)  # Lo pasamos a un array para poder operarlo de mejor manera

    Y = intercept + coef * X  # Ecuacion de la recta que se ajusta a los puntos

    plt.plot(Y)
    plt.show()  # Imprimimos la recta


if __name__ == '__main__':
    print("Hola")

