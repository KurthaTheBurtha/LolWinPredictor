from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def lolwin():
    # print(lol_set.DESCR)

    inputs = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=range(2, 40), dtype=np.float32)
    target = np.genfromtxt('high_diamond_ranked_10min.csv', delimiter=',', skip_header=1, usecols=[1], dtype=np.int8)
    inputs_train, inputs_test, target_train, target_test = train_test_split(inputs,target,test_size=0.2,random_state=42)


    # accuracies = []
    # models = []

    # for i in range (1,32):
    #     models.append(MLPClassifier(random_state=0, batch_size=i,max_iter=1000))
    # print()
    # for model in models:
    #     model.fit(inputs_train,target_train)
    #     accuracy = model.score(inputs_test,target_test)
    #     accuracies.append(accuracy)
    #
    # print(max(accuracies))
    # plot_model_accuracies(models,accuracies)
    #train data on 20% of data
    classifier = MLPClassifier(random_state=0,hidden_layer_sizes=(25,50,10),batch_size=3)
    classifier.fit(inputs_train, target_train)
    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    accuracies = cross_val_score(classifier, inputs, target, cv=kf)
    print(accuracies)
    print(np.mean(accuracies))
    display_accuracy(target_test,classifier.predict(inputs_test),["negative","positive"],"Accuracies")

    print(classifier.score(inputs_test,target_test))

    # old code
    # results = classifier.predict(inputs_test)
    # print(str(round((results == target_test).mean()*100,2))+'% accuracy')

def display_accuracy(target, predictions, labels, title):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

def plot_model_accuracies(models, accuracies):
    model_names = [f"Batch Size {i}" for i in range(len(models))]
    #plot model
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracies')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)  # Set the y-axis limits to 0-1 (accuracy range)
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    lolwin()
