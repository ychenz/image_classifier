import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist as fashion_mnist
from keras.utils import np_utils
import os

from model import SimpleCNNModel

class FashionData:
    def __init__(self,training=True):
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        self.X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        self.X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
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
        self.model = SimpleCNNModel((28, 28, 1), 10,training=training)

    def visualize(self):
        plt.imshow(self.X_train.reshape(self.X_train.shape[0], 28, 28)[0])
        plt.show()

    def fit(self, epochs=20, lr=0.001):
        self.model.compile(lr=lr)
        self.model.fit(self.X_train, self.y_train, self.X_test, self.y_test, batch_size=256, epochs=epochs)

    def classify(self,x,labels):
        return self.model.classify(x,labels)

def clear_models(dir):
    for the_file in os.listdir(dir):
        file_path = os.path.join(dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # Training
    clear_models("models")
    fashion_model = FashionData(training=True)  # 90% accuracy in 3 min
    fashion_model.fit(epochs=10,lr=0.001)

    from keras.models import Sequential,load_model
    import keras

    model = Sequential()
    model = load_model("models/fashion_model")
    model.load_weights("models/fashion_model.h5")  # Added this line
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
    # default_graph = tf.get_default_graph()
    # default_graph.finalize()

    score = model.evaluate(fashion_model.X_test,fashion_model.y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    for i in range(0,len(fashion_model.y_test)):
        x = fashion_model.X_test[i]
        y = fashion_model.y_test[i].tolist()
        x_new = x.reshape(1, 28, 28, 1)
        category = model.predict_classes(x_new, verbose=1)

        print("Predicted: " + labels[category[0]])
        print("Expected: " + labels[y.index(max(y))])
        print("")
        plt.imshow(x.reshape(28, 28))
        plt.show()
