
def decision_boundaries():
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    iris = load_iris(as_frame=True)


    # now to split the data into training and test sets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    x = iris.data[["petal width (cm)"]].values
    y = iris.target_names[iris.target] == "virginica"

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    log_reg = LogisticRegression().fit(X_train, y_train)
    print("Training set score: {:.3f}".format(log_reg.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(log_reg.score(X_test, y_test)))

    x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(x_new)
    decision_boundary = x_new[y_proba[:, 1] >= 0.5][0,0]

    plt.plot(x_new, y_proba[:, 0], "b--", label="Not Virginica", linewidth=2)
    plt.plot(x_new, y_proba[:, 1], "g-", label="Virginica", linewidth=2)
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(x[y==0], y[y==0], "bs")
    plt.plot(x[y==1], y[y==1], "g^")
    plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    plt.show()

    print("Decision boundary: ", decision_boundary)

def softmaxregression():
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    iris = load_iris(as_frame=True)

    # now to split the data into training and test sets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    x = iris.data[["petal width (cm)", "petal length (cm)"]].values
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    softmax = LogisticRegression(C=30).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(softmax.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(softmax.score(X_test, y_test)))

    # now to plot the decision boundaries
    x0, x1 = np.meshgrid(
        np.linspace(0, 3.5, 500).reshape(-1, 1),
        np.linspace(0, 1.8, 200).reshape(-1, 1),
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]

    y_proba = softmax.predict_proba(x_new)
    y_predict = softmax.predict(x_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.figure(figsize=(10, 4))
    plt.plot(x[y==2, 0], x[y==2, 1], "g^", label="Iris virginica")
    plt.plot(x[y==1, 0], x[y==1, 1], "bs", label="Iris versicolor")
    plt.plot(x[y==0, 0], x[y==0, 1], "yo", label="Iris setosa")

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3.5, 0, 1.8])
    plt.show()
