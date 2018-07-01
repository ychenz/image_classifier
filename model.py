import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential,model_from_json,load_model
import matplotlib.pyplot as plt

class SimpleCNNModel:
    def __init__(self, input_shape,num_classes,training=True):
        if training:
            self.history = AccuracyHistory()

        self.model = Sequential()
        self.model.add(
            Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))  # shape: 24,24,32 (from (batch,28,28,1))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # shape: 12,12,32
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))  # shape: 8,8,64
        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # shape: 4,4,64
        print("Network output shape after 2nd pooling: " + str(self.model.output_shape))  # prints: 4,4,64
        self.model.add(Flatten())  # shape: 4*4*64
        print("Network output shape after flatten: "+str(self.model.output_shape))
        self.model.add(Dense(units=1024, activation='relu'))  # shape: 1024
        if training:
            self.model.add(Dropout(rate=0.4))

        self.model.add(Dense(units=num_classes, activation='softmax'))  # shape: 10, for output comparison

    def compile(self, lr=0.001):
        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr),
                           metrics=['accuracy'])

    def fit(self,x_train, y_train, x_test, y_test,batch_size=256, epochs=10):
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=[self.history, ],
                       )

        print("Training completed!")
        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("models/fashion_model.json", "w") as json_file:
        #     json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save("models/fashion_model")
        self.model.save_weights("models/fashion_model.h5")
        print("Saved model to disk")
        self.history.plot_acc()

    def load(self,file):
        # load weights into new model
        # json_file = open(json_file, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        self.model = load_model(file)
        print("Loaded model from disk")

    '''
        :param data: 1 channel image data with shape (28,28,1)
    '''
    def classify(self, data,labels):
        y = self.model.predict(data, batch_size=1, verbose=1)
        y = y.tolist()[0]
        print(y)
        category = y.index(max(y))
        return labels[category]

    def evaluate(self,x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

    def plot_acc(self):
        plt.plot(self.acc)
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.show()
