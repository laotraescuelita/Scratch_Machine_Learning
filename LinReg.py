#Resolver un problema de regresion lineal con un algorimto hecho por nosotros.

#Importar algunas librerias que siempre son utiles.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# De scikit learn podemos utilizar su libreria para crear una matriz con datos artificiales.
from sklearn.datasets import make_regression

matriz, vector, coeficientes = make_regression(
n_samples = 100,
n_features = 3,
n_informative = 3,
n_targets = 1,
noise = 0.0,
coef = True,
random_state = 1)

#Agregar la matriz a un dataframe.
df = pd.DataFrame( matriz , columns= ["x1","x2","x3"])
print("\nMatriz transformada a un  dataframe \n", df.head())
#Agregar la variable a clasificar
df["target"] = vector

#Mostrar la forma que tienen las variables.
fig, axes = plt.subplots(1, df.shape[1], figsize=(15, 4))
for i in range( df.shape[1]):
    axes[i].hist( df.iloc[:,i] )    
plt.show()

#Crear una clase para la regresion lineal.
class RegresionLineal:
    def __init__( self, x, y, lr, iter_): #Recibira la matriz, el vector, el learning rate y el numero de iteraciones.
        self.X = x
        self.y = y
        self.lr = lr
        self.iter = iter_        
        self.pesos = np.zeros( (self.X.shape[1], 1) ) #Inicializmos los pesos 
        self.costo = 0 #El costo comienza desde cero.
        self.lista_de_costos = [] #vamos a almacenar los costos.

    def entrenar( self ):
        m = self.X.shape[0] # Las dimensiones de la matriz.

        for i in range( self.iter): #Vamos a recorrer el circuito el numero de veces que nos indiquen.            
            # 1) resolver la ecuación Ax=b. #Multiplicar las matrices X y los pesos, eso da un vector y.
            y_pred = np.dot( self.X, self.pesos ) 
            # 2) Necesitamos la función de costo.
            self.costo = ( 1/( 2*m ) ) * np.sum( np.square( y_pred - self.y )) 
            #3) Hay que derivar la función costo con respeto a los pesos.
            dcosto_dpesos = (1/m) * np.dot(self.X.T, y_pred-self.y)
            #4) Aplicar la técnica "gradient descent" para minimizar la función de costo.
            self.pesos = self.pesos - self.lr * dcosto_dpesos
            #Guardemos los costos para graficarlos y mirar si han disminuido.
            self.lista_de_costos.append( self.costo )
            
            #Imprimir los costos para saber si el algoritmo si esta minimizando la función.
            if (i%(self.iter/10) == 0):
                print( "\nCosto después de ", i, " ietración : ", self.costo)
                

        return self.pesos, self.lista_de_costos

    def predecir( self, x):
        y_pred =  np.dot( x , self.pesos)
        return y_pred

#Separar la matriz y el vector a predecir.
X = df.iloc[:,0:3].values
y = df.iloc[:,-1].values
#Dividir una parte para entrenamiento y otra para pruebas.
X_train = X[:70]
X_test = X[70:]
y_train = y[:70]
y_test = y[70:]
#Miremos las dimensiones.
print( "\nDimensiones de Xtrain \n", X_train.shape)
print( "\nDimensiones de Xtest \n", X_test.shape)
print( "\nDimensiones de ytrain \n", y_train.shape)
print( "\nDimensiones de ytest \n", y_test.shape)

#Como he tenido con problemas con la variable a predecir la volvi un array.
y_train = np.array(y_train).reshape(y_train.shape[0],1)
y_test = np.array(y_test).reshape(y_test.shape[0],1)
print("\nDimensiones de ytrain \n", y_train.shape)
print("\nDimensiones de ytest \n", y_test.shape)

#Iniciemos los parametros.
lr = 0.001
iter_ = 1000
#Iniiciar la clase con los parametros.
reglin = RegresionLineal( X_train, y_train, lr, iter_)
#Entrenar el modelo.
pesos, costos = reglin.entrenar()
#Mirar el grafico de los costos.
rango = np.arange( 0, iter_)
plt.plot( costos, rango )
plt.show()


#Predecir
yhat = reglin.predecir(X_test) 

#Mostrar la forma que tienen las variables 
fig, axes = plt.subplots(1,2, figsize=(15,4) )
axes[0].scatter( X_test[:,1], y_test )
axes[1].scatter( X_test[:,1], yhat )
plt.show()


print("\n Utilizando los modelos lineales de scikitlearn\n")

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))
print("Resultado en entrenamiento: {:.2f}".format(lr.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(lr.score(X_test, y_test)))

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(ridge.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(ridge.score(X_test, y_test)))
print("Numero de variables utilizadas: {}".format(np.sum(ridge.coef_ != 0)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(ridge10.score(X_test, y_test)))
print("Numero de variables utilizadas:{}".format(np.sum(ridge10.coef_ != 0)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(ridge01.score(X_test, y_test)))
print("Numero de variables utilizadas: {}".format(np.sum(ridge01.coef_ != 0)))


from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(lasso.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(lasso.score(X_test, y_test)))
print("Numero de variables utilizadas: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Numero de variables utilizadas: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Resultado en pruebas: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Numero de variables utilizadas: {}".format(np.sum(lasso00001.coef_ != 0)))

