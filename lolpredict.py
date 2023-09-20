from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
import statistics

def lolwin():
    # print(lol_set.DESCR)

    lol_set = np.genfromtxt('high_diamond_ranked_10min.csv',delimiter=',' ,names = [
        'gameId','blueWins','blueWardsPlaced','blueWardsDestroyed','blueFirstBlood','blueKills','blueDeaths',
        'blueAssists','blueEliteMonsters','blueDragons','blueHeralds','blueTowersDestroyed','blueTotalGold',
        'blueAvgLevel','blueTotalExperience','blueTotalMinionsKilled','blueTotalJungleMinionsKilled',
        'blueGoldDiff','blueExperienceDiff','blueCSPerMin','blueGoldPerMin','redWardsPlaced','redWardsDestroyed',
        'redFirstBlood','redKills','redDeaths','redAssists','redEliteMonsters','redDragons','redHeralds',
        'redTowersDestroyed','redTotalGold','redAvgLevel','redTotalExperience','redTotalMinionsKilled',
        'redTotalJungleMinionsKilled','redGoldDiff','redExperienceDiff','redCSPerMin','redGoldPerMin'
    ],skip_header=True)
    # digits_set = datasets.load_digits()
    inputs = lol_set[:,[1]]
    target = lol_set[:,[0]]+lol_set[:,[2,len[lol_set]]]
    print(f'Shape of input data array:  {inputs.shape}')
    print(f'Shape of output data array: {target.shape}')

    # This is the neural network
    classifier = MLPClassifier(random_state=0)
    print()
    test_size = 10

    # Train on all the data AFTER the first 10 (i.e. on 1787 images)
    classifier.fit(inputs[test_size:], target[test_size:])

    # Test on ONLY the first 10 digits
    # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
    results = classifier.predict(inputs[:test_size])
    test = statistics.mean(results)
    real = statistics.mean(target)
    print(test*100,end='\n')
    print(real*100,end='\n')


    # Print to the terminal the results
    # for i in range(len(results)):
    #     print('Neural Net guessed: ' + str(results[i]))
    #     print('Actual value: ' + str(target[i]))
    #     img = inputs[i].reshape(8, 8)  # reshape to look like an 8x8 image
    #     display_img(img)

if __name__ == '__main__':
    lolwin()
