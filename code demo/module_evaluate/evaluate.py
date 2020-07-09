import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def draw_roc_curve(y_true, y_predict):
    # Compute ROC curve and ROC area for each class
    l_1 = []
    for e_predict in y_predict:
        if e_predict < 0:
            l_1.append(e_predict)

    # print(l_1)

    fpr, tpr, threshold = roc_curve(y_true, y_predict, pos_label=1)
    print(threshold)
    print("False positive rate: \n {}".format(fpr))
    print("True positive rate: \n {}".format(tpr))
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
