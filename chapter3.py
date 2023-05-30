def classiSDG():
    # Load MNIST dataset
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", as_frame=False)

    # Split into training and test sets
    X, y =mnist.data, mnist.target
    #print(X.shape, y.shape)
    #print(X, y)

    import matplotlib.pyplot as plt

    #image_data = X[0]
    #image = image_data.reshape(28, 28)
    #plt.imshow(image, cmap="binary")
    #plt.axis("off")
    #plt.show()

    # Split into training and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # training a binary clasifier
    y_train_5 = (y_train == '5')
    y_test_5 = (y_test == '5')

    # an SDG is better because it trains each data set indivisually
    # and is better for large data sets

    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    #sgd_clf.fit(X_train, y_train_5)

    #print(sgd_clf.predict([X[0]])) # the image 5 is at index 0

    # performance measures
    #from sklearn.model_selection import cross_val_score
    #james = cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring="accuracy")
    #print(james)

    #just like cross validation score. the predict function uses cross validation
    # to predict the accuracy of the model
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64)) # this is a float64 because the scaler requires it
    # y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=2)
    # trying to save time

    sgd_clf.fit(X_train_scaled, y_train)
    y_train_pred = sgd_clf.predict(X_train_scaled)

    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
    plt.show()

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(y_train_5, y_train_pred))

    # negative | positive
    # true negative | false positive
    # false negative | true positive

    #precision
    # tp/ (tp + fp)
    # tp = number of true positives
    # fp = number of false positives

    #recall
    # sensitivity of the classifier or true positive rate
    # recall = tp / (tp + fn)
    # fn = number of false negatives

    from sklearn.metrics import precision_score, recall_score, f1_score
    #print(precision_score(y_train_5, y_train_pred))
    #print(recall_score(y_train_5, y_train_pred))
    #print(f1_score(y_train_5, y_train_pred))

    #lets try this one more time
    #from sklearn.metrics import precision_recall_curve
    #y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    #precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    #def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
     #   plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
     #   plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
     #   plt.xlabel("Threshold")
     #   plt.legend(loc="upper left")
     #   plt.ylim([0, 1])
     #   plt.show()

    #plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    #plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
    #plt.xlabel("Recall")
    #plt.ylabel("Precision")
    #plt.show()


    #Error AnLysis
    #confusion matrix
    #precision/recall tradeoff






