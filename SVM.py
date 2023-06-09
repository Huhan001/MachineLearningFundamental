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
    print(poly_kernel_svm_clf.predict([[1.5, 0.5]]))



