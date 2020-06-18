# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_summary import DataFrameSummary

import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

import warnings

# --------------------------------------------------------------------------------------
# Semilla
SEED = 150
np.random.seed(SEED)

# Clase que funciona como cualquier estimador
class ClfSwitcher(BaseEstimator):
    def __init__(
        self,
        estimator = LogisticRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

def PlotGridSearchResults(results, title=""):
    plt.figure(figsize=(13, 13))
    plt.title(title, fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

# Lectura de los datos de entrenamiento
datos = pd.read_csv("./datos/OnlineNewsPopularity.csv", delimiter = ', ', engine = 'python')
# Quitamos los atributos no predictivos
datos = datos.drop(columns=['url', 'timedelta'])
print(datos)

# Datos perdidos
datos_perdidos = datos.columns[datos.isnull().any()]
print(len(datos_perdidos))
datos_perdidos = datos.columns[datos.isna().any()]
print(len(datos_perdidos))

y = datos.iloc[:, -1]
X = datos.iloc[:, :-1]

y = y.apply(lambda x: -1.0 if x < 1400 else 1.0)

print("Valor mínimo de las caraterísticas del conjunto de datos: {}".format(X.values.min()))
print("Valor máximo de las caraterísticas del conjunto de datos: {}".format(X.values.max()))

# Vemos si las clases estan bien balanceadas
y_df = pd.DataFrame(data = y)
numero_elementos = []
clases = [1.0,-1.0]
for i in clases:
    numero_elementos.append(y_df['shares'].value_counts()[i])

df_plot = pd.DataFrame(columns= ["Clases", "Número de ejemplos"], data =[[c,n] for c, n in zip(clases,numero_elementos)])
sns.barplot(x="Clases", y ="Número de ejemplos", data = df_plot)
plt.title("Número de ejemplos de cada clase en el conjunto de datos")
plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

# Preprocesado
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95))]

preprocesador = Pipeline(preprocesado)

# Mostramos la matriz de correlaciones antes del preprocesado de datos
def mostrar_correlaciones(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = datos.corr()
    sns.heatmap(corr,
        mask=np.zeros_like(corr, dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax)
    f.suptitle('Matriz Correlaciones')
    plt.show()

mostrar_correlaciones(X_train)
input("\n--- Pulsar tecla para continuar ---\n")

# Mostramos la matriz de correlaciones después del preprocesado de datos
def muestra_correlaciones_procesados(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = np.corrcoef(datos.T)
    sns.heatmap(corr,
        mask=np.zeros_like(corr, dtype=np.bool),
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax)
    f.suptitle('Matriz Correlaciones')
    plt.show()

datos_preprocesados = preprocesador.fit_transform(X_train)
muestra_correlaciones_procesados(datos_preprocesados)
input("\n--- Pulsar tecla para continuar ---\n")

# Entrenamiento
# Añadimos el clasificador ClfSwitcher para evitar errores de compilación
preprocesado = [("escalado", StandardScaler()),
                ("PCA", PCA(n_components=0.95)), ('clf', ClfSwitcher())]

preprocesador = Pipeline(preprocesado)

# Modelos
modelos = [
    {'clf': [LogisticRegression(penalty='l2', # Regularización Ridge (L2)
        solver='lbfgs', # Algoritmo a utilizar en el problema de optimización, aunque es el dado por defecto
        max_iter=1000)],
        'clf__C':[2.0, 1.0, 0.1, 0.01, 0.001]},
    {'clf': [SVC(kernel='rbf', # kernel gausiano
        class_weight="balanced", # clases balanceadas
        random_state=SEED)],
        'clf__C': [10**a for a in range(-4, 2)]},
    {'clf': [RandomForestClassifier(random_state=SEED,
                                    class_weight="balanced")],
        'clf__max_depth': [10, 20, 30, 40, 50],
        'clf__n_estimators': [50, 100, 150, 200]},
]

# cross-validation
grid = GridSearchCV(preprocesador, modelos, scoring='accuracy', cv=5, n_jobs=-1)

grid.fit(X_train, y_train)
results = grid.cv_results_
PlotGridSearchResults(results, "GridSearchCV sobre LogisticRegression, SVM y RandomForestClassifier con distintos parametros")
clasificador = grid.best_estimator_

# Mostramos el clasificador elegido
print("Clasificador elegido: {}".format(clasificador))
y_predict = clasificador.predict(X_test)

# Matriz de confusion
cm = confusion_matrix(y_test, y_predict)
cm = 100*cm.astype("float64")/cm.sum(axis=1)[:,np.newaxis]
fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(cm, cmap ="BuGn")
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set(title="Matriz de confusión",
    xticks=np.arange(2),
    yticks=np.arange(2),
    xlabel="Etiqueta real",
    ylabel="Etiqueta predicha")

# Añadimos los porcentajes a las celdas
for i in range(2):
    for j in range(2):
        ax.text(j, i, "{:.0f}%".format(cm[i, j]), ha="center", va="center")

plt.show()
input("\n--- Pulsar tecla para continuar ---\n")

# Resultados
print("E_in: {}".format(1 - clasificador.score(X_train, y_train)))
print("E_test: {}".format(1 - clasificador.score(X_test, y_test)))
