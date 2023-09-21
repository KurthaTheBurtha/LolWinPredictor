from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np


def lolwin():
    # print(lol_set.DESCR)

    inputs = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=range(2, 40), dtype=np.float32)
    target = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=[1], dtype=np.int8)

    # This is the neural network
    classifier = MLPClassifier(random_state=0)
    print()
    test_size = 2000

    #train data on test_size data
    classifier.fit(resample(target,n_samples=test_size), resample(inputs,n_samples=test_size))

    results = classifier.predict(inputs[:test_size])
    print((results == target[:test_size]).mean())
    print(resample(target,n_samples=10))
    print(resample(inputs,n_samples=10))
    # print(np.random.choice(inputs,2000))


if __name__ == '__main__':
    lolwin()
