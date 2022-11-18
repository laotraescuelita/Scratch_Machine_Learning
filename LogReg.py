#Resolver un problema de regresion lineal con un algorimto hecho por nosotros.

#Importar algunas librerias que siempre son utiles.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# De scikit learn podemos utilizar su libreria para crear una matriz con datos artificiales.
from sklearn.datasets import make_classification
# Generar la matriz con variables numericas y el vector a predecir
matriz, vector = make_classification(n_samples = 100,
n_features = 3,
n_informative = 3,
n_redundant = 0,
n_classes = 2,
weights = [.25, .75],
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


#Crear una clase para la regresion logistica.
class RegresionLogistica:
    def __init__( self, x, y, lr, iter_): #Recibira la matriz, el vector, learnig rate y el numero de iteraciones.
        self.X = x 
        self.y = y 
        self.lr = lr 
        self.iter = iter_
        self.lista_de_costos = []
        self.costo = 0        
        self.m = self.X.shape[0] #Numero de filas
        self.n = self.X.shape[1] #Numero de columnas
        self.pesos = np.zeros( (self.n,1)) #Iniciar los pesos con ceros.
        self.sesgo = 0  #iniciar el sesgo en cero
    
    
    def sigmoid(self, x ): #Funcion de activación.
        return 1 / ( 1 + np.exp(-x) )
    
    def entrenar( self ):
        #Entrenar el numero de veces que se indique.
        for i in range( self.iter ):
            #1) Multiplicar matrices.
            z = np.dot( self.X, self.pesos ) + self.sesgo
            #2) Los resultados de z se pasan por la función de activación, eso los coloca entre -1 y 1. 
            a = self.sigmoid( z )            
            #3) Necesitamos una función de costo para después minmizarla.
            self.costo = -( 1/ self.m ) * np.sum( self.y * np.log( a ) + (1 - self.y ) * np.log (1 - a) )
            #4) derivadas parciales
            #Con respecto al peso.
            dcosto_dpesos = ( 1 / self.m ) * np.dot( self.X.T , a - self.y )
            #Con respecto al sesgo.
            dcosto_dsesgo = ( 1 / self.m ) * np.sum( a - self.y )            
            #5) La técnica del "gradient descent" nos ayudara a minmizar la función de costo.
            self.pesos = self.pesos - lr * dcosto_dpesos
            self.sesgo = self.sesgo - lr * dcosto_dsesgo
                        
            #Almacenar lso costos para graficar su compartamiento.
            self.lista_de_costos.append( self.costo )            
            #Imprimir los costos para mirar si va reduciendo su compartamiento.
            if (i%(self.iter/10) == 0):
                print( "Cost0 después de ", i, " ietración : ", self.costo)
                
        return self.pesos, self.sesgo, self.lista_de_costos
    
    def predecir(self, x):
        yhat = np.dot( x , self.pesos)
        for i in range(len(yhat)):
            if yhat[i] < 0.5:
                yhat[i] = 0
            else:
                yhat[i] = 1
        return yhat


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

#Pasemos los parametros 
lr = .01
iter_ = 1000
reglog = RegresionLogistica(X_train, y_train, lr, iter_)
pesos, sesgo, costos = reglog.entrenar()
#Graficar el compartamiento de los costos.
iter_range = np.arange( iter_ )
plt.plot( iter_range, costos )
plt.show()

#Utilizemos el test set para predecir 
yhat = reglog.predecir(X_test)
print( "El modelo predijo de manera correcta : ", np.sum( yhat == y_test ) / y_test.shape[0], "%")


#Utilizemos la libreria sklearn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(X_train, y_train)
print("Resultado en entrenamiento: {:.3f}".format(logreg.score(X_train, y_train)))
print("Resultado en pruebas:: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Resultado en pruebas: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Resultado en entrenamiento: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Resultado en pruebas: {:.3f}".format(logreg001.score(X_test, y_test)))


