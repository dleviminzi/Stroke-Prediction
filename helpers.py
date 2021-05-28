import itertools

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# Helper function that allows you to draw nicely formatted confusion matrices
def draw_confusion_matrix(y, yhat, classes):
    '''
        Draws a confusion matrix for the given target and predictions
        Adapted from scikit-learn and discussion example.
    '''
    plt.cla()
    plt.clf()
    matrix = confusion_matrix(y, yhat)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    num_classes = len(classes)
    plt.xticks(np.arange(num_classes), classes, rotation=90)
    plt.yticks(np.arange(num_classes), classes)
    
    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def model_report_from(model, X_test, y_test, y_test_pred):
    if len(y_test_pred) != len(X_test):
        y_test_pred = model.predict(X_test)
        
    print(f"accuracy = {1 - sum(abs(y_test_pred - y_test))/len(y_test)}")
    print(f"precision = {precision_score(y_test, y_test_pred)}")
    print(f"recall = {recall_score(y_test, y_test_pred)}")
    print(f"f1 = {f1_score(y_test, y_test_pred)}")
    print(f"ROC AUC score = {roc_auc_score(y_test, y_test_pred)}")

    # display confusion matrix
    draw_confusion_matrix(y_test, y_test_pred, ["NO STROKE", "STROKE"])

    # draw ROC curve
    plot_roc_curve(model, X_test, y_test)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

