def classification():
    from sklearn.datasets import load_iris
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    # Load the iris dataset

    iris = load_iris(as_frame=True)
    x = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y = (iris.target == 2)

    # Create a pipeline and fit the model
    svm_clf = make_pipeline(StandardScaler(),
                            LinearSVC(C=1, loss="hinge"))

    svm_clf.fit(x, y)

    # Predict the class of a new flower with petal length 5cm and petal width 2cm
    new_flower = [[5, 2], [4, 1]]
    print(svm_clf.predict(new_flower))
    print(svm_clf.decision_function(new_flower))


def make_moons():
    from sklearn.datasets import make_moons
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    x, y = make_moons(n_samples=100, noise=0.15)
    poly_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, max_iter=10000, random_state=42))
    ])
    poly_svm_clf.fit(x, y)
    print(poly_svm_clf.predict([[1.5, 0.5]]))

    # Polynomial Kernel
    from sklearn.svm import SVC
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(x, y)

    # Generate a meshgrid of points to evaluate the model over
    import numpy as np
    import matplotlib.pyplot as plt

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the labels for all points in the meshgrid
    Z = poly_kernel_svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and support vectors
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(poly_kernel_svm_clf.named_steps['svc'].support_vectors_[:, 0],
                poly_kernel_svm_clf.named_steps['svc'].support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()


def make_moons_rbf():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC

    x, y = make_moons(n_samples=100, noise=0.15)

    poly_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, max_iter=10000, random_state=42))
    ])
    poly_svm_clf.fit(x, y)

    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(x, y)

    # Generate a meshgrid of points to evaluate the model over
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the labels for all points in the meshgrid
    Z = poly_kernel_svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and support vectors
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.scatter(poly_kernel_svm_clf.named_steps['svm_clf'].support_vectors_[:, 0],
                poly_kernel_svm_clf.named_steps['svm_clf'].support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()