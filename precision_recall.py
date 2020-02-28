import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

# Hay 1797 digitos representados en imagenes 8x8
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
dd = target!=1
target[dd]= 0

# Vamos a hacer un split training test en la mitad
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#PCA sobre todo train
covTrain = np.cov(x_train.T)
valoresTrain, vectoresTrain = np.linalg.eig(covTrain)
valoresTrain = np.real(valoresTrain)
vectoresTrain = np.real(vectoresTrain)
ii = np.argsort(-valoresTrain)
valoresTrain = valoresTrain[ii]
vectoresTrain = vectoresTrain[:,ii]

# Vamos a entrenar solamente con los digitos iguales a 0
numero = 0
dd = y_train==numero
covZeros = np.cov(x_train[dd].T)
valoresZeros, vectoresZeros = np.linalg.eig(covZeros)
valoresZeros = np.real(valoresZeros)
vectoresZeros = np.real(vectoresZeros)
ii = np.argsort(-valoresZeros)
valoresZeros = valoresZeros[ii]
vectoresZeros = vectoresZeros[:,ii]


# Vamos a entrenar solamente con los digitos iguales a 1
numero = 1
dd = y_train==numero
covUnos = np.cov(x_train[dd].T)
valoresUnos, vectoresUnos = np.linalg.eig(covUnos)
valoresUnos = np.real(valoresUnos)
vectoresUnos = np.real(vectoresUnos)
ii = np.argsort(-valoresUnos)
valoresUnos = valoresUnos[ii]
vectoresUnos = vectoresUnos[:,ii]


#Producto matricial entre todos los datos Uno
ResultadosX_train_Unos = x_train@vectoresUnos
ResultadosX_test_Unos = x_test@vectoresUnos
#Producto matricial entre todos los datos Zeros
ResultadosX_train_Zeros = x_train@vectoresZeros
ResultadosX_test_Zeros = x_test@vectoresZeros
#Producto matricial entre todos los datos Train
ResultadosX_train = x_train@vectoresTrain
ResultadosX_test = x_test@vectoresTrain
    
def fiteando(ResultadosX_train,ResultadosX_test,y_train,y_test,true):

    clf = LinearDiscriminantAnalysis()
    clf.fit(ResultadosX_train[:,0:10],y_train)
    probs_test = clf.predict_proba(ResultadosX_test[:,0:10])[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, probs_test, pos_label=true)
    F1_test = 2 * (precision * recall) / (precision + recall)
    F1_test = F1_test[1:]
    return F1_test, precision, recall, thresholds

F1_test_Unos, precision_Unos, recall_Unos, probs_Unos = fiteando(ResultadosX_train_Unos,ResultadosX_test_Unos,y_train,y_test,1)
F1_test_Zeros, precision_Zeros, recall_Zeros, probs_Zeros = fiteando(ResultadosX_train_Zeros,ResultadosX_test_Zeros,y_train,y_test,1)
F1_test, precision, recall, probs = fiteando(ResultadosX_train,ResultadosX_test,y_train,y_test,1)


dd = np.argmax(F1_test)
dd1 = np.argmax(F1_test_Unos)
dd0 = np.argmax(F1_test_Zeros)
plt.figure(figsize=(12, 12))

plt.subplot(1,2,1)
plt.plot(probs,F1_test, label = 'Sobre Todos' )
plt.plot(probs_Unos,F1_test_Unos, label = 'Sobre los 1' )
plt.plot(probs_Zeros,F1_test_Zeros, label = 'Sobre los 0' )
plt.scatter(probs_Unos[dd1],F1_test_Unos[dd1], c= 'r')
plt.scatter(probs[dd],F1_test[dd], c= 'r')
plt.scatter(probs_Zeros[dd0],F1_test_Zeros[dd0], c= 'r')
plt.legend()
plt.title('F1')
plt.xlabel('Probabilidad')
plt.ylabel('F1')


plt.subplot(1,2,2)
plt.plot(recall, precision, label = 'Sobre Todos')
plt.plot(recall_Unos, precision_Unos, label = 'Sobre los 1')
plt.plot(recall_Zeros, precision_Zeros, label = 'Sobre los 0')
plt.legend()
plt.title('PR')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.savefig('F1_prec_recall.png')