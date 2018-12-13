
# AV2 - Redes Convolucionais

### Alunos

- Tiago Silva
- Yasmin Romi

## Overview

***Redes Neurais Convolucionais***: Utilizada amplamente no reconhecimento de imagens, uma rede neural convolucional (CNN do inglês Convolutional Neural network ou ConvNet) é uma classe de rede neural artificial do tipo feed-forward, que vem sendo aplicada com sucesso no processamento e análise de imagens digitais.

Iremos dividir em duas etapas onde serão aplicadas técnicas diferentes em cima do mesmo conjunto de dados. Devermos abordar esse problema de duas maneiras:

- Criando uma rede neural completamente conectacada de uma única camada oculta e com uma camada de saída de duas unidades com softmax
- Uma rede convolucional

# Rede Completamente Conectada

### Questões

**Neurônios na camada escondida ?**

64 na camada oculta
2 neurônios na camada de Saída

**Quantas Camadas?**

Utilizaremos Duas Camadas, sendo uma camada oculta e uma camada de saída.

**Com Regularização? Qual?**

Não.
Uma forma de lidar com o overfitting pra nossa rede completamente conectada seria adicionar uma regularização a ela. No entanto, isso faria com que tivessemos que adicionais mais camadas a rede e deixaríamos de ter a arquitetura pedida pelo trabalho, portanto, manteremo-as como apresentada acima.

**Qual a melhor taxa de aprendizado?**

**Qual a função de ativação usada ?**

Na Camada de Saída, utilizamos a *softmax*
Na Camada escondida, utlizaremos a *relu*

## Preparações

Faremos a implementação utilizando o [Keras](https://www.tensorflow.org/guide/keras), uma API para modelagem de redes neurais que roda por cima dos principais frameworks de redes neurais. No nosso caso, o framework utilizado como base será o TensowFlow. 


```python
# Tensoflow and keras
import tensorflow as tf
from keras import backend as K
import keras

# Utilities
import numpy as np
import h5py
import random as rn
import os

# Comando necessário para que o código tenha reproducibilidade
# para certas operações hash-based
os.environ['PYTHONHASHSEED'] = '0'

# Comando necessário para inicializar valores randômicos gerados pelo Numpy
np.random.seed(42)

# Comando necessário para inicializar valores randômicos gerados pelo Python
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)



def load_dataset():
    train_dataset = h5py.File('./../data/train_catvnoncat.h5', 'r')
    
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('./../data/test_catvnoncat.h5', 'r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(-1, train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(-1, test_set_y_orig.shape[0])

    return (train_set_x_orig, train_set_y_orig, test_set_x_orig,
            test_set_y_orig, classes)


# Carregando os datasets de treinamento e teste
train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
train_set_y = train_set_y.reshape(-1, 1)
test_set_y = test_set_y.reshape(-1, 1)
```

    Using TensorFlow backend.



```python
from keras.utils import to_categorical

def preprocessing_neural(x, y):
    # Função para aplicar pré-processamento nos vetores
    # de features e target

    # Achatando o número de dimensões do vetor de features
    # Isso significa que estamos transformando nossa matriz de imagens de 64x64 num vetor
    # no formato (nSamples, 64x64) que será servido para a rede neural como uma única feature
    x_flatten = x.reshape(x.shape[0], -1)

    # Normalizando os valores de RGB para ficarem entre 0 e 1
    x_flatten = x_flatten.astype('float32')
    x_flatten = x_flatten/255

    # Transformando os valores inteiros de y em categorias
    # por ser o formato necessário para o Keras realizar a
    # classificação de múltiplas classes
    y_cat = to_categorical(y)

    return x_flatten, y_cat


# Aplicando pré-processamento nos vetores de features
train_set_x, train_set_y = preprocessing_neural(train_set_x, train_set_y)
test_set_x, test_set_y = preprocessing_neural(test_set_x, test_set_y)
print(train_set_x.shape)
print(train_set_y.shape)
input_shape = train_set_x.shape[1:]
print(input_shape)
```

    (209, 12288)
    (209, 2)
    (12288,)


Agora que temos nossos conjuntos de dados carregados e pré-processado, o próximo passo é a modelagem da rede neural completamente conectada. Essa rede deverá conter apenas uma camada oculta e uma camada de saída com 2 neurônios usando a função de ativação `softmax`. 

Por conta disso, a nível de arquitetura da rede, só temos dois parâmetros para escolher: a quantidade de neurônios e a função de ativação da camada oculta.

A escolha do número de neurônios se deu através de testes baseado em exemplos de redes encontradas na internet que visam executar um trabalho semelhante.

A função de ativação utilizada foi a `relu` pois tem sido a função de ativação padrão para esse tipo de rede atualmente por ser a que apresenta melhores resultados.

Em relação ao otimizador, os testes foram feitos principalmente utilizando o `rmsprop` e o `sgd`.


```python
from keras.models import Sequential
from keras.layers import Dense, Flatten

def create_model_neural(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(2, activation='softmax'))
    return model


def train_model_neural(model, x_train, y_train, x_test, y_test, batch, epoch):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch,
                        validation_data=(x_test, y_test), shuffle=False)
    return history

# Criando o modelo de rede neural para predição
batch_size = 15
epochs = 100
model = create_model_neural(input_shape)
hist_neural = train_model_neural(model, train_set_x, train_set_y,
                                 test_set_x, test_set_y,
                                 batch_size, epochs)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 64)                786496    
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 130       
    =================================================================
    Total params: 786,626
    Trainable params: 786,626
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 209 samples, validate on 50 samples
    Epoch 1/100
    209/209 [==============================] - 0s 2ms/step - loss: 5.2208 - acc: 0.6268 - val_loss: 10.6379 - val_acc: 0.3400
    Epoch 2/100
    209/209 [==============================] - 0s 949us/step - loss: 6.0351 - acc: 0.5407 - val_loss: 10.6328 - val_acc: 0.3400
    Epoch 3/100
    209/209 [==============================] - 0s 1ms/step - loss: 3.5341 - acc: 0.5550 - val_loss: 5.6479 - val_acc: 0.3400
    Epoch 4/100
    209/209 [==============================] - 0s 869us/step - loss: 1.3583 - acc: 0.6411 - val_loss: 2.4212 - val_acc: 0.3800
    Epoch 5/100
    209/209 [==============================] - 0s 990us/step - loss: 2.2699 - acc: 0.5789 - val_loss: 4.5054 - val_acc: 0.3400
    Epoch 6/100
    209/209 [==============================] - 0s 958us/step - loss: 1.2693 - acc: 0.6842 - val_loss: 2.0816 - val_acc: 0.3600
    Epoch 7/100
    209/209 [==============================] - 0s 950us/step - loss: 1.9688 - acc: 0.5885 - val_loss: 3.5507 - val_acc: 0.3400
    Epoch 8/100
    209/209 [==============================] - 0s 1ms/step - loss: 1.1767 - acc: 0.6555 - val_loss: 3.9563 - val_acc: 0.3400
    Epoch 9/100
    209/209 [==============================] - 0s 907us/step - loss: 1.2211 - acc: 0.6411 - val_loss: 4.8097 - val_acc: 0.3400
    Epoch 10/100
    209/209 [==============================] - 0s 873us/step - loss: 1.1901 - acc: 0.6746 - val_loss: 4.0016 - val_acc: 0.3400
    Epoch 11/100
    209/209 [==============================] - 0s 945us/step - loss: 1.2304 - acc: 0.6411 - val_loss: 4.7591 - val_acc: 0.3400
    Epoch 12/100
    209/209 [==============================] - 0s 921us/step - loss: 1.6207 - acc: 0.6220 - val_loss: 2.6938 - val_acc: 0.3200
    Epoch 13/100
    209/209 [==============================] - 0s 935us/step - loss: 0.9799 - acc: 0.7225 - val_loss: 2.7591 - val_acc: 0.3400
    Epoch 14/100
    209/209 [==============================] - 0s 931us/step - loss: 0.8548 - acc: 0.7033 - val_loss: 1.9884 - val_acc: 0.3400
    Epoch 15/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.8879 - acc: 0.7081 - val_loss: 2.7337 - val_acc: 0.3200
    Epoch 16/100
    209/209 [==============================] - 0s 978us/step - loss: 0.9390 - acc: 0.6890 - val_loss: 3.1749 - val_acc: 0.3200
    Epoch 17/100
    209/209 [==============================] - 0s 993us/step - loss: 0.9684 - acc: 0.6746 - val_loss: 2.3611 - val_acc: 0.3200
    Epoch 18/100
    209/209 [==============================] - ETA: 0s - loss: 1.1548 - acc: 0.671 - 0s 1ms/step - loss: 1.0957 - acc: 0.6794 - val_loss: 1.0976 - val_acc: 0.5400
    Epoch 19/100
    209/209 [==============================] - 0s 921us/step - loss: 0.9572 - acc: 0.6938 - val_loss: 1.1230 - val_acc: 0.4600
    Epoch 20/100
    209/209 [==============================] - 0s 878us/step - loss: 0.8289 - acc: 0.7033 - val_loss: 1.1189 - val_acc: 0.4400
    Epoch 21/100
    209/209 [==============================] - 0s 921us/step - loss: 0.8577 - acc: 0.7177 - val_loss: 1.1137 - val_acc: 0.4600
    Epoch 22/100
    209/209 [==============================] - 0s 926us/step - loss: 0.7636 - acc: 0.7129 - val_loss: 1.1736 - val_acc: 0.4400
    Epoch 23/100
    209/209 [==============================] - 0s 921us/step - loss: 0.7647 - acc: 0.7512 - val_loss: 1.1197 - val_acc: 0.4600
    Epoch 24/100
    209/209 [==============================] - 0s 911us/step - loss: 0.7257 - acc: 0.7321 - val_loss: 1.0871 - val_acc: 0.4800
    Epoch 25/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.7513 - acc: 0.7129 - val_loss: 1.0902 - val_acc: 0.4800
    Epoch 26/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.6597 - acc: 0.7464 - val_loss: 1.1701 - val_acc: 0.4800
    Epoch 27/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.6332 - acc: 0.7751 - val_loss: 1.0753 - val_acc: 0.4800
    Epoch 28/100
    209/209 [==============================] - 0s 988us/step - loss: 0.6752 - acc: 0.7656 - val_loss: 1.0938 - val_acc: 0.4800
    Epoch 29/100
    209/209 [==============================] - 0s 868us/step - loss: 0.3849 - acc: 0.8325 - val_loss: 1.1298 - val_acc: 0.5000
    Epoch 30/100
    209/209 [==============================] - 0s 878us/step - loss: 0.6715 - acc: 0.7751 - val_loss: 1.1599 - val_acc: 0.5200
    Epoch 31/100
    209/209 [==============================] - 0s 926us/step - loss: 0.6541 - acc: 0.7608 - val_loss: 1.1077 - val_acc: 0.5000
    Epoch 32/100
    209/209 [==============================] - 0s 881us/step - loss: 0.4041 - acc: 0.8278 - val_loss: 3.2138 - val_acc: 0.3200
    Epoch 33/100
    209/209 [==============================] - 0s 866us/step - loss: 0.5972 - acc: 0.7943 - val_loss: 1.0699 - val_acc: 0.5600
    Epoch 34/100
    209/209 [==============================] - 0s 859us/step - loss: 0.7055 - acc: 0.7464 - val_loss: 1.6207 - val_acc: 0.3600
    Epoch 35/100
    209/209 [==============================] - 0s 940us/step - loss: 0.4339 - acc: 0.8182 - val_loss: 1.0741 - val_acc: 0.5600
    Epoch 36/100
    209/209 [==============================] - 0s 926us/step - loss: 0.6229 - acc: 0.7751 - val_loss: 1.3130 - val_acc: 0.4200
    Epoch 37/100
    209/209 [==============================] - 0s 902us/step - loss: 0.3721 - acc: 0.8182 - val_loss: 0.9440 - val_acc: 0.5600
    Epoch 38/100
    209/209 [==============================] - 0s 849us/step - loss: 0.4754 - acc: 0.7990 - val_loss: 1.1419 - val_acc: 0.5200
    Epoch 39/100
    209/209 [==============================] - 0s 859us/step - loss: 0.4448 - acc: 0.8756 - val_loss: 1.6750 - val_acc: 0.4400
    Epoch 40/100
    209/209 [==============================] - 0s 869us/step - loss: 0.4559 - acc: 0.8182 - val_loss: 1.1150 - val_acc: 0.5600
    Epoch 41/100
    209/209 [==============================] - 0s 926us/step - loss: 0.3852 - acc: 0.8278 - val_loss: 1.0649 - val_acc: 0.5600
    Epoch 42/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.3721 - acc: 0.8517 - val_loss: 1.1229 - val_acc: 0.5600
    Epoch 43/100
    209/209 [==============================] - 0s 897us/step - loss: 0.3494 - acc: 0.8660 - val_loss: 1.0710 - val_acc: 0.5600
    Epoch 44/100
    209/209 [==============================] - 0s 840us/step - loss: 0.3693 - acc: 0.8660 - val_loss: 1.5234 - val_acc: 0.4600
    Epoch 45/100
    209/209 [==============================] - 0s 892us/step - loss: 0.3943 - acc: 0.8373 - val_loss: 1.3425 - val_acc: 0.4800
    Epoch 46/100
    209/209 [==============================] - 0s 948us/step - loss: 0.5621 - acc: 0.7895 - val_loss: 1.7729 - val_acc: 0.3800
    Epoch 47/100
    209/209 [==============================] - 0s 941us/step - loss: 0.3797 - acc: 0.8469 - val_loss: 1.9033 - val_acc: 0.3800
    Epoch 48/100
    209/209 [==============================] - 0s 919us/step - loss: 0.3821 - acc: 0.8469 - val_loss: 1.5029 - val_acc: 0.4200
    Epoch 49/100
    209/209 [==============================] - 0s 873us/step - loss: 0.4310 - acc: 0.8325 - val_loss: 1.3761 - val_acc: 0.4600
    Epoch 50/100
    209/209 [==============================] - 0s 969us/step - loss: 0.3506 - acc: 0.8517 - val_loss: 1.3189 - val_acc: 0.4400
    Epoch 51/100
    209/209 [==============================] - 0s 868us/step - loss: 0.3265 - acc: 0.8612 - val_loss: 1.1310 - val_acc: 0.5200
    Epoch 52/100
    209/209 [==============================] - 0s 931us/step - loss: 0.3465 - acc: 0.8469 - val_loss: 2.8549 - val_acc: 0.3800
    Epoch 53/100
    209/209 [==============================] - 0s 916us/step - loss: 0.3599 - acc: 0.8565 - val_loss: 2.1336 - val_acc: 0.3600
    Epoch 54/100
    209/209 [==============================] - 0s 930us/step - loss: 0.2549 - acc: 0.8947 - val_loss: 0.9036 - val_acc: 0.6400
    Epoch 55/100
    209/209 [==============================] - 0s 904us/step - loss: 0.3477 - acc: 0.8373 - val_loss: 1.6271 - val_acc: 0.3400
    Epoch 56/100
    209/209 [==============================] - 0s 892us/step - loss: 0.2595 - acc: 0.9043 - val_loss: 0.9834 - val_acc: 0.6400
    Epoch 57/100
    209/209 [==============================] - 0s 935us/step - loss: 0.3577 - acc: 0.8325 - val_loss: 1.1434 - val_acc: 0.5200
    Epoch 58/100
    209/209 [==============================] - 0s 921us/step - loss: 0.2281 - acc: 0.9043 - val_loss: 1.9212 - val_acc: 0.3800
    Epoch 59/100
    209/209 [==============================] - 0s 997us/step - loss: 0.3873 - acc: 0.8660 - val_loss: 2.3542 - val_acc: 0.3800
    Epoch 60/100
    209/209 [==============================] - 0s 927us/step - loss: 0.2844 - acc: 0.8900 - val_loss: 1.3978 - val_acc: 0.4800
    Epoch 61/100
    209/209 [==============================] - 0s 888us/step - loss: 0.2371 - acc: 0.9091 - val_loss: 1.0932 - val_acc: 0.5400
    Epoch 62/100
    209/209 [==============================] - 0s 939us/step - loss: 0.3798 - acc: 0.8804 - val_loss: 0.9761 - val_acc: 0.6400
    Epoch 63/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.2520 - acc: 0.8900 - val_loss: 1.1331 - val_acc: 0.5400
    Epoch 64/100
    209/209 [==============================] - 0s 902us/step - loss: 0.2098 - acc: 0.9091 - val_loss: 0.7239 - val_acc: 0.7200
    Epoch 65/100
    209/209 [==============================] - 0s 902us/step - loss: 0.3382 - acc: 0.8756 - val_loss: 3.1320 - val_acc: 0.3800
    Epoch 66/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.2283 - acc: 0.9282 - val_loss: 1.7756 - val_acc: 0.4200
    Epoch 67/100
    209/209 [==============================] - 0s 935us/step - loss: 0.1568 - acc: 0.9330 - val_loss: 0.7161 - val_acc: 0.7600
    Epoch 68/100
    209/209 [==============================] - 0s 931us/step - loss: 0.2583 - acc: 0.8947 - val_loss: 2.9053 - val_acc: 0.3600
    Epoch 69/100
    209/209 [==============================] - 0s 943us/step - loss: 0.2177 - acc: 0.9234 - val_loss: 1.3424 - val_acc: 0.6000
    Epoch 70/100
    209/209 [==============================] - 0s 940us/step - loss: 0.3284 - acc: 0.8852 - val_loss: 1.1510 - val_acc: 0.6600
    Epoch 71/100
    209/209 [==============================] - 0s 947us/step - loss: 0.1360 - acc: 0.9474 - val_loss: 1.1219 - val_acc: 0.6600
    Epoch 72/100
    209/209 [==============================] - 0s 911us/step - loss: 0.2778 - acc: 0.9187 - val_loss: 1.2282 - val_acc: 0.6200
    Epoch 73/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.1459 - acc: 0.9426 - val_loss: 0.9793 - val_acc: 0.7000
    Epoch 74/100
    209/209 [==============================] - 0s 955us/step - loss: 0.3191 - acc: 0.8900 - val_loss: 1.3057 - val_acc: 0.5800
    Epoch 75/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.1666 - acc: 0.9474 - val_loss: 3.1004 - val_acc: 0.3600
    Epoch 76/100
    209/209 [==============================] - 0s 969us/step - loss: 0.2477 - acc: 0.9091 - val_loss: 0.9702 - val_acc: 0.7200
    Epoch 77/100
    209/209 [==============================] - 0s 983us/step - loss: 0.1605 - acc: 0.9330 - val_loss: 1.8003 - val_acc: 0.5400
    Epoch 78/100
    209/209 [==============================] - 0s 926us/step - loss: 0.5081 - acc: 0.8708 - val_loss: 1.1763 - val_acc: 0.7000
    Epoch 79/100
    209/209 [==============================] - 0s 882us/step - loss: 0.1338 - acc: 0.9617 - val_loss: 2.6970 - val_acc: 0.4400
    Epoch 80/100
    209/209 [==============================] - 0s 859us/step - loss: 0.1849 - acc: 0.9330 - val_loss: 3.5597 - val_acc: 0.3400
    Epoch 81/100
    209/209 [==============================] - 0s 878us/step - loss: 0.1601 - acc: 0.9522 - val_loss: 1.7147 - val_acc: 0.5600
    Epoch 82/100
    209/209 [==============================] - 0s 878us/step - loss: 0.1238 - acc: 0.9617 - val_loss: 4.0904 - val_acc: 0.3400
    Epoch 83/100
    209/209 [==============================] - 0s 868us/step - loss: 0.1847 - acc: 0.9378 - val_loss: 1.0024 - val_acc: 0.7800
    Epoch 84/100
    209/209 [==============================] - 0s 902us/step - loss: 0.2799 - acc: 0.8995 - val_loss: 1.0804 - val_acc: 0.7200
    Epoch 85/100
    209/209 [==============================] - 0s 869us/step - loss: 0.1886 - acc: 0.9282 - val_loss: 1.9258 - val_acc: 0.5600
    Epoch 86/100
    209/209 [==============================] - 0s 921us/step - loss: 0.1780 - acc: 0.9234 - val_loss: 1.1917 - val_acc: 0.6800
    Epoch 87/100
    209/209 [==============================] - 0s 1ms/step - loss: 0.1127 - acc: 0.9617 - val_loss: 2.9296 - val_acc: 0.3800
    Epoch 88/100
    209/209 [==============================] - 0s 988us/step - loss: 0.1508 - acc: 0.9139 - val_loss: 1.3239 - val_acc: 0.7000
    Epoch 89/100
    209/209 [==============================] - 0s 955us/step - loss: 0.1032 - acc: 0.9713 - val_loss: 1.6858 - val_acc: 0.5800
    Epoch 90/100
    209/209 [==============================] - 0s 859us/step - loss: 0.1142 - acc: 0.9617 - val_loss: 1.1695 - val_acc: 0.7400
    Epoch 91/100
    209/209 [==============================] - 0s 869us/step - loss: 0.3070 - acc: 0.8708 - val_loss: 2.0856 - val_acc: 0.5200
    Epoch 92/100
    209/209 [==============================] - 0s 864us/step - loss: 0.1238 - acc: 0.9522 - val_loss: 2.0634 - val_acc: 0.5600
    Epoch 93/100
    209/209 [==============================] - 0s 894us/step - loss: 0.1647 - acc: 0.9474 - val_loss: 3.0953 - val_acc: 0.4000
    Epoch 94/100
    209/209 [==============================] - 0s 878us/step - loss: 0.0925 - acc: 0.9713 - val_loss: 1.3614 - val_acc: 0.7000
    Epoch 95/100
    209/209 [==============================] - 0s 864us/step - loss: 0.1430 - acc: 0.9522 - val_loss: 1.4052 - val_acc: 0.7000
    Epoch 96/100
    209/209 [==============================] - 0s 945us/step - loss: 0.2149 - acc: 0.9187 - val_loss: 1.8967 - val_acc: 0.5800
    Epoch 97/100
    209/209 [==============================] - 0s 911us/step - loss: 0.0727 - acc: 0.9761 - val_loss: 1.2972 - val_acc: 0.7800
    Epoch 98/100
    209/209 [==============================] - 0s 959us/step - loss: 0.0543 - acc: 0.9904 - val_loss: 1.6335 - val_acc: 0.6200
    Epoch 99/100
    209/209 [==============================] - 0s 945us/step - loss: 0.1824 - acc: 0.9330 - val_loss: 1.7481 - val_acc: 0.6000
    Epoch 100/100
    209/209 [==============================] - 0s 869us/step - loss: 0.2356 - acc: 0.8804 - val_loss: 1.2115 - val_acc: 0.7000


Abaixo iremos avaliar a performance da nossa rede completamente conectada através dos valores de perda (*loss*) e acurácia (*accuracy*).


```python
import matplotlib.pyplot as plt

%matplotlib inline

def evaluate_model(model, x, y):
    score = model.evaluate(x, y)
    print('Loss value: {}'.format(score[0]))
    print('Accuracy: {}'.format(score[1]))


def plot_accuracy(history):
    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()
    

evaluate_model(model, test_set_x, test_set_y)
plot_accuracy(hist_neural)
```

    50/50 [==============================] - 0s 140us/step
    Loss value: 1.2115476655960082
    Accuracy: 0.7000000071525574



![png](AV2_files/AV2_7_1.png)



![png](AV2_files/AV2_7_2.png)


Olhando unicamente para os números, nossa perda no conjunto de validação foi de **1.21** e nossa acurácia foi de 0.70, o que não é ruim. No entanto, ao observar os gráficos plotados podemos tirar algumas conclusões mais precisas sobre a nossa rede.

Tanto no gráfico de perda quanto no de acurácia, as linhas do conjunto de treinamento e validação estão bastante distantes umas das outras. Além disso, no gráfico de perda podemos ver inicialmente uma queda no erro no conjunto de validação (até próximo da *epoch* 20) mas depois o erro se estabiliza e passa a ter picos de subida ao invés de seguir uma curva descendente como no conjunto de treinamento. Esses são indícios de que nossa rede está sofrente de *overfitting*, ou seja, o modelo "decorou" o conjunto de treinamento, mas não vai bem quando apresentamos dados não vistos antes por ele (durante o treinamento).

# Rede Convulacional

Agora vamos modelar e treinar nossa rede convolucional. Iremos realizar novamente o processo de leitura e pré-processamento dos dados. 

O de leitura para garantir que estamos trabalhando com o conjunto de dados base e o pré-processamento pois ele é diferente para a rede convolucional.

### Questões

**Neurônios na camada escondida ?

256 na camada oculta
2 neurônios na camada de Saída

**Quantas Camadas?

Utilizaremos Duas Camadas, sendo uma camada oculta e uma camada de saída (completamente conectadas).

**Com Regularização? Qual?

Sim.
Utilizamos Dropout e Pooling.

**Qual a melhor taxa de aprendizado?

**Qual a função de ativação usada ?

Na Camada de Saída, utilizamos a *softmax*
Na Camada escondida, utlizaremos a *relu*


```python
# Leitura do conjunto de dados de treinamento
train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
train_set_y = train_set_y.reshape(-1, 1)
test_set_y = test_set_y.reshape(-1, 1)
```


```python
def preprocessing_cnn(x, y):
    # Função para aplicar pré-processamento nos vetores
    # de features e target

    # Normalizando os valores de RGB para ficarem entre 0 e 1
    x_flatten = x.astype('float32')
    x_flatten = x_flatten/255

    # Transformando os valores inteiros de y em categorias
    # por ser o formato necessário para o Keras realizar a
    # classificação de múltiplas classes
    y_cat = to_categorical(y)

    return x_flatten, y_cat

# Aplicando pré-processamento nos vetores de features
train_set_x, train_set_y = preprocessing_cnn(train_set_x, train_set_y)
test_set_x, test_set_y = preprocessing_cnn(test_set_x, test_set_y)
print(train_set_x.shape)
print(train_set_y.shape)
input_shape = train_set_x[0].shape
print(input_shape)
```

    (209, 64, 64, 3)
    (209, 2)
    (64, 64, 3)


Tendo os dados carregados e pré-processados, podemos modelar e treinar a rede convolucional. A processo de modelagem dela e de encontrar os hiperparâmetros que teriam um melhor resultado foi mais complexo por não ter sido dada indicação do formato da rede como foi feito para a rede completamente conectada. Mais uma vez, a abordagem utilizada foi a de buscar exemplos na internet que atacavam problemas semelhantes e, à partir deles, ir fazendo alterações na arquitetura até que fosse atingido um resultado satisfatório.


```python
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def create_model_cnn(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
              input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model


def train_model_cnn(model, x_train, y_train, x_test, y_test, batch, epoch):
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=batch, epochs=epoch,
                        validation_data=(x_test, y_test), shuffle=False)
    return history

# Criando o modelo de rede neural para predição
batch_size = 10
epochs = 100
model = create_model_cnn(input_shape)
hist_cnn = train_model_cnn(model, train_set_x, train_set_y,
                           test_set_x, test_set_y,
                           batch_size, epochs)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 64, 64, 64)        1792      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 21, 21, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 21, 21, 64)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 10, 10, 128)       0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 10, 10, 128)       0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 12800)             0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 256)               3277056   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 2)                 514       
    =================================================================
    Total params: 3,353,218
    Trainable params: 3,353,218
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 209 samples, validate on 50 samples
    Epoch 1/100
    209/209 [==============================] - 5s 26ms/step - loss: 0.6933 - acc: 0.6172 - val_loss: 0.9152 - val_acc: 0.3400
    Epoch 2/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.6554 - acc: 0.6555 - val_loss: 0.8256 - val_acc: 0.3400
    Epoch 3/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.6569 - acc: 0.6220 - val_loss: 0.7881 - val_acc: 0.3400
    Epoch 4/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.6181 - acc: 0.6364 - val_loss: 0.7801 - val_acc: 0.3400
    Epoch 5/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.6140 - acc: 0.6364 - val_loss: 0.7771 - val_acc: 0.3400
    Epoch 6/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.6242 - acc: 0.6603 - val_loss: 0.7840 - val_acc: 0.3400
    Epoch 7/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.6043 - acc: 0.6603 - val_loss: 0.7621 - val_acc: 0.3400
    Epoch 8/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5967 - acc: 0.6651 - val_loss: 0.7560 - val_acc: 0.3400
    Epoch 9/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5576 - acc: 0.6746 - val_loss: 0.8024 - val_acc: 0.3400
    Epoch 10/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5534 - acc: 0.6890 - val_loss: 0.7842 - val_acc: 0.3400
    Epoch 11/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5526 - acc: 0.6842 - val_loss: 0.8246 - val_acc: 0.3400
    Epoch 12/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5253 - acc: 0.7033 - val_loss: 0.7687 - val_acc: 0.3400
    Epoch 13/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.5139 - acc: 0.7225 - val_loss: 0.8238 - val_acc: 0.3400
    Epoch 14/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4985 - acc: 0.7656 - val_loss: 0.7579 - val_acc: 0.3400
    Epoch 15/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4999 - acc: 0.7464 - val_loss: 0.7092 - val_acc: 0.4600
    Epoch 16/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4895 - acc: 0.7560 - val_loss: 0.7375 - val_acc: 0.4400
    Epoch 17/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.4811 - acc: 0.7703 - val_loss: 0.7365 - val_acc: 0.4000
    Epoch 18/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4576 - acc: 0.7608 - val_loss: 0.5898 - val_acc: 0.7000
    Epoch 19/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4478 - acc: 0.7703 - val_loss: 0.6043 - val_acc: 0.6800
    Epoch 20/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.4413 - acc: 0.7847 - val_loss: 0.7478 - val_acc: 0.3800
    Epoch 21/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4347 - acc: 0.7799 - val_loss: 0.5646 - val_acc: 0.7200
    Epoch 22/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4243 - acc: 0.7847 - val_loss: 0.5957 - val_acc: 0.7000
    Epoch 23/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4213 - acc: 0.7703 - val_loss: 0.5783 - val_acc: 0.7200
    Epoch 24/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4114 - acc: 0.7943 - val_loss: 0.6836 - val_acc: 0.6200
    Epoch 25/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3838 - acc: 0.8278 - val_loss: 0.6987 - val_acc: 0.5400
    Epoch 26/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4123 - acc: 0.7990 - val_loss: 0.4780 - val_acc: 0.8400
    Epoch 27/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3810 - acc: 0.8373 - val_loss: 0.6752 - val_acc: 0.6400
    Epoch 28/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.3747 - acc: 0.8325 - val_loss: 0.9783 - val_acc: 0.3600
    Epoch 29/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.4013 - acc: 0.8182 - val_loss: 0.5201 - val_acc: 0.7200
    Epoch 30/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3802 - acc: 0.8373 - val_loss: 0.6056 - val_acc: 0.6800
    Epoch 31/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3676 - acc: 0.8325 - val_loss: 0.5509 - val_acc: 0.7000
    Epoch 32/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3635 - acc: 0.8278 - val_loss: 0.5311 - val_acc: 0.7200
    Epoch 33/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3628 - acc: 0.8278 - val_loss: 0.5253 - val_acc: 0.7200
    Epoch 34/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3947 - acc: 0.8086 - val_loss: 0.4873 - val_acc: 0.8200
    Epoch 35/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3812 - acc: 0.8278 - val_loss: 0.4711 - val_acc: 0.7600
    Epoch 36/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3516 - acc: 0.8421 - val_loss: 0.6572 - val_acc: 0.6200
    Epoch 37/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3576 - acc: 0.8421 - val_loss: 0.6450 - val_acc: 0.6400
    Epoch 38/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3510 - acc: 0.8421 - val_loss: 0.4333 - val_acc: 0.8400
    Epoch 39/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3581 - acc: 0.8373 - val_loss: 0.4607 - val_acc: 0.8400
    Epoch 40/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3373 - acc: 0.8469 - val_loss: 0.4164 - val_acc: 0.8400
    Epoch 41/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3399 - acc: 0.8421 - val_loss: 0.4899 - val_acc: 0.8200
    Epoch 42/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3494 - acc: 0.8517 - val_loss: 0.4762 - val_acc: 0.8200
    Epoch 43/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3296 - acc: 0.8804 - val_loss: 0.4354 - val_acc: 0.8400
    Epoch 44/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3186 - acc: 0.8565 - val_loss: 0.5017 - val_acc: 0.7400
    Epoch 45/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3166 - acc: 0.8565 - val_loss: 0.4418 - val_acc: 0.8200
    Epoch 46/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3082 - acc: 0.8660 - val_loss: 0.4169 - val_acc: 0.8600
    Epoch 47/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3189 - acc: 0.8565 - val_loss: 0.6268 - val_acc: 0.6600
    Epoch 48/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3262 - acc: 0.8612 - val_loss: 0.7667 - val_acc: 0.5800
    Epoch 49/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3390 - acc: 0.8517 - val_loss: 0.4686 - val_acc: 0.8400
    Epoch 50/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.3103 - acc: 0.8852 - val_loss: 0.4016 - val_acc: 0.8800
    Epoch 51/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.3227 - acc: 0.8612 - val_loss: 0.4522 - val_acc: 0.8200
    Epoch 52/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.3176 - acc: 0.8660 - val_loss: 0.4094 - val_acc: 0.9000
    Epoch 53/100
    209/209 [==============================] - 6s 28ms/step - loss: 0.3008 - acc: 0.8660 - val_loss: 0.4282 - val_acc: 0.8600
    Epoch 54/100
    209/209 [==============================] - 5s 26ms/step - loss: 0.3121 - acc: 0.8708 - val_loss: 0.4821 - val_acc: 0.8400
    Epoch 55/100
    209/209 [==============================] - 5s 25ms/step - loss: 0.3125 - acc: 0.8660 - val_loss: 0.4124 - val_acc: 0.9000
    Epoch 56/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.3015 - acc: 0.8660 - val_loss: 0.5195 - val_acc: 0.7600
    Epoch 57/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.3001 - acc: 0.8708 - val_loss: 0.4402 - val_acc: 0.8400
    Epoch 58/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2911 - acc: 0.8660 - val_loss: 0.3871 - val_acc: 0.9000
    Epoch 59/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2938 - acc: 0.8708 - val_loss: 0.4090 - val_acc: 0.8800
    Epoch 60/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2890 - acc: 0.8660 - val_loss: 0.3785 - val_acc: 0.9000
    Epoch 61/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.3002 - acc: 0.8612 - val_loss: 0.4267 - val_acc: 0.8800
    Epoch 62/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2680 - acc: 0.8708 - val_loss: 0.4463 - val_acc: 0.8600
    Epoch 63/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2873 - acc: 0.8756 - val_loss: 0.3795 - val_acc: 0.9000
    Epoch 64/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2830 - acc: 0.8804 - val_loss: 0.4121 - val_acc: 0.8800
    Epoch 65/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2563 - acc: 0.8995 - val_loss: 0.3708 - val_acc: 0.9000
    Epoch 66/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2841 - acc: 0.8900 - val_loss: 0.3638 - val_acc: 0.8800
    Epoch 67/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2709 - acc: 0.8804 - val_loss: 0.3783 - val_acc: 0.9000
    Epoch 68/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2830 - acc: 0.8756 - val_loss: 0.4602 - val_acc: 0.8200
    Epoch 69/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2564 - acc: 0.8900 - val_loss: 0.5132 - val_acc: 0.7600
    Epoch 70/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.2646 - acc: 0.8900 - val_loss: 0.4336 - val_acc: 0.8400
    Epoch 71/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2472 - acc: 0.8947 - val_loss: 0.3837 - val_acc: 0.8800
    Epoch 72/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2469 - acc: 0.8852 - val_loss: 0.3919 - val_acc: 0.8800
    Epoch 73/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2771 - acc: 0.8708 - val_loss: 0.3856 - val_acc: 0.9200
    Epoch 74/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2421 - acc: 0.9043 - val_loss: 0.3823 - val_acc: 0.8800
    Epoch 75/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2530 - acc: 0.8995 - val_loss: 0.3710 - val_acc: 0.8800
    Epoch 76/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2342 - acc: 0.8947 - val_loss: 0.3456 - val_acc: 0.9200
    Epoch 77/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2392 - acc: 0.8900 - val_loss: 0.3662 - val_acc: 0.9000
    Epoch 78/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2284 - acc: 0.9139 - val_loss: 0.4080 - val_acc: 0.8600
    Epoch 79/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2334 - acc: 0.9043 - val_loss: 0.3728 - val_acc: 0.8800
    Epoch 80/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.2283 - acc: 0.8900 - val_loss: 0.3975 - val_acc: 0.9000
    Epoch 81/100
    209/209 [==============================] - 6s 26ms/step - loss: 0.2284 - acc: 0.9187 - val_loss: 0.3357 - val_acc: 0.9200
    Epoch 82/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2214 - acc: 0.9139 - val_loss: 0.3563 - val_acc: 0.9000
    Epoch 83/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2040 - acc: 0.9187 - val_loss: 0.3287 - val_acc: 0.9200
    Epoch 84/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2390 - acc: 0.8804 - val_loss: 0.3682 - val_acc: 0.9000
    Epoch 85/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2015 - acc: 0.9330 - val_loss: 0.3518 - val_acc: 0.9200
    Epoch 86/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2007 - acc: 0.9282 - val_loss: 0.3940 - val_acc: 0.9000
    Epoch 87/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2105 - acc: 0.9043 - val_loss: 0.3728 - val_acc: 0.9200
    Epoch 88/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2086 - acc: 0.9187 - val_loss: 0.3791 - val_acc: 0.9000
    Epoch 89/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2022 - acc: 0.9234 - val_loss: 0.3767 - val_acc: 0.9000
    Epoch 90/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2060 - acc: 0.9091 - val_loss: 0.3444 - val_acc: 0.9200
    Epoch 91/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2005 - acc: 0.9187 - val_loss: 0.3433 - val_acc: 0.9200
    Epoch 92/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2065 - acc: 0.9234 - val_loss: 0.3323 - val_acc: 0.9400
    Epoch 93/100
    209/209 [==============================] - 5s 24ms/step - loss: 0.1878 - acc: 0.9139 - val_loss: 0.3500 - val_acc: 0.9000
    Epoch 94/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.1797 - acc: 0.9330 - val_loss: 0.3399 - val_acc: 0.9200
    Epoch 95/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.1797 - acc: 0.9139 - val_loss: 0.3535 - val_acc: 0.9400
    Epoch 96/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.1753 - acc: 0.9426 - val_loss: 0.3365 - val_acc: 0.9200
    Epoch 97/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.1955 - acc: 0.9330 - val_loss: 0.3345 - val_acc: 0.9200
    Epoch 98/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.2043 - acc: 0.9139 - val_loss: 0.3484 - val_acc: 0.9400
    Epoch 99/100
    209/209 [==============================] - 5s 22ms/step - loss: 0.2000 - acc: 0.9234 - val_loss: 0.3421 - val_acc: 0.9400
    Epoch 100/100
    209/209 [==============================] - 5s 23ms/step - loss: 0.1724 - acc: 0.9378 - val_loss: 0.3450 - val_acc: 0.9400


A rede convolucional modelada possuí uma arquitetura de 10 camadas, dividas da seguinte forma:

- 2 camadas convolucionais de 2 dimensões
- 2 camadas de pooling (utiliando Max Pooling)
- 3 camadas de dropout
- 1 camada Flatten
- 2 camadas densas (completamente conectadas)

As camadas convolucionais são, como o próprio nome nos dá a entender, as principais camadas desse tipo de rede. Nessas camadas o que acontece, resumidamente, é um aprendizado de filtros por parte da rede, que irá percorrer a entrada que lhe for dada e irá "ativar" quando encontrar algum padrão que esteja de acordo com os filtros aprendidos.

Geralmente entre camadas convolucionais são adicionadas camada de pooling. Essas camadas tem como função a diminuição progressiva do tamanho espacial para que se tenham menos parâmetros e computações sendo processadas na rede. Outra utilidade dessa camada é o controle de *overfitting* que, como dito anteriormente, acontece quando uma rede "decora" os parâmetros em que foi treinada e não conseguir fazer uma generalização para um outro conjunto de dados que venha a ser apresentado a ela.

Outra maneira de combater o *overfitting* é através da adição de camadas de *dropout*. Nessas camadas algumas entradas são selecionados randômicamente para que sejam desligadas da rede, o que torna mais difícil que a rede que está sendo treinada fique totalmente aderente a um conjunto de treinamento.

A camada *Flatten* server para adequar os dados produzidos nas camadas de convolução para serem servidos para as camadas completamente conectadas, que são as responsáveis por produzir a classificação que será a saída do modelo. Ela funciona da mesma forma que numa rede completamente conectada, aonde temos que realizar essa operação com o mesmo objetivo.

Como dito anteriormente, tanto essa arquitetura quanto os parãmetros de cada camada foram descobertos à partir de modelos encontrados na internet que visavam resolver problemas de classificação semelhantes ao nosso. Foram feitos uma série de testes manuais com diferentes combinações de parâmetros em cada camada para chegar até o modelo final. Vamos ver qual foi o resultado desse modelo.


```python
evaluate_model(model, test_set_x, test_set_y)
plot_accuracy(hist_cnn)
```

    50/50 [==============================] - 0s 5ms/step
    Loss value: 0.34498459339141846
    Accuracy: 0.9399999904632569



![png](AV2_files/AV2_15_1.png)



![png](AV2_files/AV2_15_2.png)


# Conclusões

### Rede Completamente Conectada

**Loss**: 1.2
**Accuracy**: 0.7

### Rede Covulacional:

**Loss**: 0.3
**Accuracy**: 0.9

A rede convolucional apresenta uma performance muito melhor do que a rede neural completamente conectada que treinamos na etapa anterior desse trabalho, atingindo uma perda de 0.34 e uma acurácia de praticamente 94%. 

Olhando para o gráfico da acurácia podemos ver que a rede generalizou bem, tendo desempenho muito semelhante no conjunto de treinamento e validação. No entanto, o gráfico de erro nos mostra uma certa distância entre os dois conjuntos de dados. 

Poderíamos tentar diminuir essa distância através de aplicações de técnicas como  aumento dos dados (*data augmentation*), mas a fim de simplificação, optamos por não utilizá-los.

## Referências

- [Rede Neuural Convolucional](https://pt.wikipedia.org/wiki/Rede_neural_convolucional)
- [Documentação do Keras](https://keras.io/)
- [Repositório do HDF5](https://github.com/h5py/h5py)
- [CS231n - Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Learn OpenCV](https://www.learnopencv.com/)


```python

```
