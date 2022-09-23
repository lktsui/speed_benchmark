import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from timeit import default_timer as timer

class Classifier(object):

    def __init__(self):
        self.parameters = None
        self.model = None

    def train(self, train_signals, train_labels):
        self.model.fit(train_signals, train_labels)

    def inference(self, test_ds):
        pass

    def get_acc_metrics(self, train_signals, train_labels, test_signals, test_labels):


        start = timer()
        predict_train = self.model.predict(train_signals)
        end = timer()


        time_per_pt = (end-start)/188

        predict_test = self.model.predict(test_signals)

        len_train = len(predict_train)
        len_test = len(predict_test)

        correct_train = 0.0
        correct_test = 0.0

        for i in range(0, len_train):
            if predict_train[i] == train_labels[i]:
                correct_train += 1.0

        for i in range(0, len_test):
            if predict_test[i] == test_labels[i]:
                correct_test += 1.0

        cm_train = confusion_matrix(train_labels, predict_train)
        cm_test = confusion_matrix(test_labels, predict_test)

        return cm_train, cm_test, time_per_pt

class LogisticRegressionClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(class_weight='balanced', verbose=0, tol=1e-6,
                                        multi_class='multinomial', solver='saga',
                                        max_iter=1e6
                                            )

class RF_Classifier(Classifier):

    def __init__(self, rf_size=500):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=rf_size)

class SVM_Classifier(Classifier):

    def __init__(self):
        super().__init__()
        self.model = SVC(kernel='rbf', gamma='scale')

class NB_Classifier(Classifier):
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()

class KN_Classifier(Classifier):
    def __init__(self, neighbor_size = 5):
        super().__init__()
        self.model = KNeighborsClassifier(n_neighbors=neighbor_size)

class KerasANN_Classifier(Classifier):

    def __init__(self, input_size, af ='sigmoid', hl_size = 10):
        super().__init__()
        self.epochs = 50000
        self.patience = 5
        self.hl_size = hl_size
        self.num_gases = 4

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=input_size))
        self.model.add(tf.keras.layers.Dense(self.hl_size, activation=af))
        self.model.add(tf.keras.layers.Dense(self.num_gases, activation='softmax'))


        #compile the model using appropriate optimizer and loss
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    def train(self, train_signals, train_labels):
        # train the model
        self.model.fit(x=train_signals,
                       y=train_labels,
                       verbose=0,
                       epochs=self.epochs,
                       callbacks=[self.early_stop],
                       shuffle=True,
                       batch_size=8)

    def get_acc_metrics(self, train_signals, train_labels, test_signals, test_labels):
        network_out_train = self.model.predict(train_signals)
        network_out_test = self.model.predict(test_signals)

        start = timer()
        predict_train = self.model.predict(train_signals)
        end = timer()

        time_per_pt = (end-start)/188
        predict_test = self.model.predict(test_signals)


        predict_train = []
        predict_test = []
        for index, entry in enumerate(network_out_train):
            predicted_entry = np.argmax(entry)
            predict_train.append(predicted_entry)

        for index, entry in enumerate(network_out_test):
            predicted_entry = np.argmax(entry)
            predict_test.append(predicted_entry)

        len_train = len(predict_train)
        len_test = len(predict_test)

        correct_train = 0.0
        correct_test = 0.0

        for i in range(0, len_train):
            if predict_train[i] == train_labels[i]:
                correct_train += 1.0

        for i in range(0, len_test):
            if predict_test[i] == test_labels[i]:
                correct_test += 1.0

        cm_train = confusion_matrix(train_labels, predict_train)
        cm_test = confusion_matrix(test_labels, predict_test)

        return cm_train, cm_test, time_per_pt
