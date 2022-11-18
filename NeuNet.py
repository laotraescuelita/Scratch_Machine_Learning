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

#Funciones de activacion.
def tanh( x ):
    return np.tanh( x )

def relu( x ):
    return np.maximum( x, 0 )

def sigmoid( x ):
    return 1/( 1 + np.exp(-x))

def softmax( x ):
    expx = np.exp( x )
    return expx/ np.sum( expx, axis = 0 )
    #return expx/ np.sum( expx)

#Derivar las funciones de activación.
def derivada_tanh( x ):
    return ( 1 - np.power( x, 2 ))

def derivada_relu( x ):
    return np.array( x > 0, dtype=np.float32)

def derivada_sigmoid( x ):
    return sigmoid( x ) * ( 1 - sigmoid( x ))


#Crear una clase para la red neuronal.
class RedNeuronal:
    #Recibe la matriz, el vector, learnig rate, iteraciones, el numero de inputs, hidden layers y outputs.
    def __init__( self, x, y, lr, iter_, nx, nh, ny): 
        self.X = x 
        self.y = y 
        self.lr = lr 
        self.iter = iter_
        self.lista_de_costos = []
        self.nx = nx 
        self.nh = nh 
        self.ny = ny
        self.costo = 0 
    
    #Inicializar pesos y sesgo.
    def iniciar_pesos_sesgo( self ):
        pesos1 = np.random.rand( self.nh, self.nx)
        sesgo1 = np.random.rand( self.nh,1 )
        pesos2 = np.random.rand( self.ny, self.nh )
        sesgo2 = np.random.rand( self.ny,1 )

        #Devolver los resultados en un diccionario.
        pesos_sesgos = {
            "pesos1":pesos1,
            "sesgo1":sesgo1,
            "pesos2":pesos2,
            "sesgo2":sesgo2
        }

        return pesos_sesgos

    def propagacion_hacia_adelante(self, pesos_sesgos):
        #1)Multiplicar las matrices y sumar el sesgo.
        z1 = np.dot( pesos_sesgos["pesos1"], self.X) + pesos_sesgos["sesgo1"]
        #2) Los resultados los pasamos por la función de activación.
        a1 = relu( z1 )
        #Multiplicamos las matrices resultantes y le sumamos su sesgo.
        z2 = np.dot( pesos_sesgos["pesos2"], a1) + pesos_sesgos["sesgo2"]
        #Al resultado final o outpt le aplicamos otra funcion de activacion dependiendo del caso, este es binario
        #por eso palicamos la funcion sigmoide.
        a2 = sigmoid( z2 )

        hacia_adelante = {
        "z1":z1,
        "a1":a1,
        "z2":z2,
        "a2":a2
        }

        return hacia_adelante

    
    #Una vez que hemos obteido las predicciones de forward propagatin hay que medirals para minmizar el costo
    def propagacion_hacia_atras( self, pesos_sesgos, hacia_adelante):
        pesos1 = pesos_sesgos["pesos1"]
        sesgo1 = pesos_sesgos["sesgo1"]
        pesos2 = pesos_sesgos["pesos2"]
        sesgo2 = pesos_sesgos["sesgo2"]
        a1 = hacia_adelante["a1"]
        a2 = hacia_adelante["a2"]

        m = self.X.shape[1]
        #Necesitamos las derivadas parciales de la función costo para poder aplicar el "gradiant descent".
        dz2 = ( a2 - self.y )
        dpesos2 = (1/m)*np.dot(dz2,a1.T)
        dsesgo2 = (1/m)*np.sum(dz2, axis=1, keepdims=True)

        dz1 = (1/m)*np.dot(pesos2.T, dz2)*derivada_relu(a1)
        dpesos1 = (1/m)*np.dot(dz1,self.X.T)
        dsesgo1 = (1/m)*np.sum(dz1, axis=1, keepdims=True)

        hacia_atras = {
        "dpesos1":dpesos1,
        "dsesgo1":dsesgo1,
        "dpesos2":dpesos2,
        "dsesgo2":dsesgo2
        }

        return hacia_atras
    
    def actualizar_pesos_sesgos(self, pesos_sesgos, hacia_atras):
        pesos1 = pesos_sesgos["pesos1"]
        sesgo1 = pesos_sesgos["sesgo1"]
        pesos2 = pesos_sesgos["pesos2"]
        sesgo2 = pesos_sesgos["sesgo2"]

        pesos1 = pesos1 - self.lr * hacia_atras["dpesos1"]
        sesgo1 = sesgo1 - self.lr * hacia_atras["dsesgo1"]
        pesos2 = pesos2 - self.lr * hacia_atras["dpesos2"]
        sesgo2 = sesgo2 - self.lr * hacia_atras["dsesgo2"]

        pesos_sesgos = {
            "pesos1":pesos1,
            "sesgo1":sesgo1,
            "pesos2":pesos2,
            "sesgo2":sesgo2
        }

        return pesos_sesgos
    
    def funcion_de_costo(self, a2 ):
        m = self.y.shape[0]
        #cost = -(1/m)*np.sum( y*np.log( a2 ))
        self.cost = -(1/m)*np.sum( self.y*np.log(a2) + (1-self.y)*np.log(1-a2))
        
        return self.cost


    def entrenar( self ):
        
        pesos_sesgos_iniciales = self.iniciar_pesos_sesgo()

        for i in range( self.iter ):
            adelante = self.propagacion_hacia_adelante(pesos_sesgos_iniciales)
            costo = self.funcion_de_costo(adelante["a2"])
            atras = self.propagacion_hacia_atras(pesos_sesgos_iniciales, adelante) 
            pesos_sesgos_finales =  self.actualizar_pesos_sesgos( pesos_sesgos_iniciales, atras)

            self.lista_de_costos.append( costo )

            if( i%(iter_/10) == 0):
                print( "Costo después: ", i, " iteraciones es: ", costo )

        return pesos_sesgos_finales, adelante, self.lista_de_costos

    def predecir( self, a2 ):
        return np.argmax( a2 , 0) 

    def exactitud(self, y_hat, y):        
        return np.sum( y_hat == y) / y.size

    
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
lr = .0001
iter_ = 200
nx = X_train.shape[1]
ny = y_train.shape[1]
nh = 2


redneu = RedNeuronal(X_train.T, y_train.T, lr, iter_, nx, nh, ny)
pesos_sesgos, hacia_adelante, costos = redneu.entrenar()
#Visualizar los costos.
iter_range = np.arange( iter_ )
plt.plot( iter_range, costos )
plt.show()

y_hat = redneu.predecir(hacia_adelante["a2"])
exactitud =  redneu.exactitud(y_hat, y_test)
print( "\nLos valores del vector a predecir\n", y_hat )
print("\nEl porcentaje de exactitud en los resultados\n", exactitud )


#Utilizemos la libreria sklearn
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

mlp = MLPClassifier(random_state=0, hidden_layer_sizes=[10])
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))

