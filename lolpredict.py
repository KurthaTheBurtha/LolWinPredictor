from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np


def lolwin():
    # print(lol_set.DESCR)

    inputs = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=range(2, 40), dtype=np.float32)
    target = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=[1], dtype=np.int8)

    # This is the neural network
    classifier = MLPClassifier(random_state=0,hidden_layer_sizes=(25,50,10),batch_size=16)
    print()
    test_size = 2000

    inputs_train, inputs_test, target_train, target_test = train_test_split(inputs,target,test_size=0.2,random_state=42)
    #train data on test_size data
    classifier.fit(inputs_train, target_train)

    results = classifier.predict(inputs_test)
    print(str(round((results == target_test).mean()*100,2))+'% accuracy')



if __name__ == '__main__':
    lolwin()
