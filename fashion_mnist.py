from keras.datasets import fashion_mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from models import SimpleCNNModel
import numpy as np

class FashionData:
    def __init__(self,lr,epochs):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        self.X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        self.X_test = X_test.reshape(X_test.shape[0],28, 28, 1)
        self.y_train = np_utils.to_categorical(y_train, 10)
        self.y_test = np_utils.to_categorical(y_test, 10)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print("Input data shapes:")
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        self.model = SimpleCNNModel((28, 28, 1),10,lr=lr,epochs=epochs)

    def visualize(self):
        plt.imshow(self.X_train.reshape(self.X_train.shape[0], 28, 28)[0])
        plt.show()

    def fit(self):
        self.model.train(self.X_train,self.y_train,self.X_test,self.y_test)

if __name__ == '__main__':
    # for lr in [0.01,0.001,0.0001]:
    #     print("")
    #     print("lr: " + str(lr))
    #     print("")
    #     data = FashionData(lr=lr,epochs=3)
    #     data.fit()
    data = FashionData(lr=0.001, epochs=20)  # 90% accuracy in 3 min
    data.fit()
