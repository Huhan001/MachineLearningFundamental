def example():
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, mpl_toolkits.mplot3d
    from sklearn.model_selection import train_test_split  # splits data into training and test data
    from sklearn.pipeline import make_pipeline  # makes a pipeline
    from sklearn.preprocessing import StandardScaler  # scales data to standard normal distribution
    from sklearn.decomposition import PCA  # principal component analysis
    from sklearn.linear_model import LinearRegression  # linear regression

    # 1. Import data as a Pandas DataFrame and preprocess them for scikit-learn:
    df = pd.read_csv("/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv", sep='\t')  # loads CSV file as a Pandas dataframe
    features = ["Sunshine duration [h/day]", "Outdoor temperature [°C]", "Solar yield [kWh/day]", "Solar pump [h/day]",
                "Valve [h/day]"]
    target = "Gas consumption [kWh/day]"  # dependent variable
    X = np.c_[df[features]]  # extracts feature values as a matrix
    y = np.c_[df[target]]  # extracts target values as a one-column matrix

    # 2. Choose by random 30 \% of data as test data, i.e., 70 \% as training data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # 3. Fit and predict with a pipeline of scaling, PCA, and linear regression:
    pipe = make_pipeline(StandardScaler(), PCA(2), LinearRegression())
    pipe.fit(X_train, y_train)

    # 4. Print model score:
    print("score (train values): ", f"{pipe.score(X_train, y_train):.2%}")
    print("score (test values):", f"{pipe.score(X_test, y_test):.2%}")

    # 5. Plot 3D scatter plot:
    from mpl_toolkits.mplot3d import Axes3D

    # 5.1. Choose PCA model from pipeline and project data onto the principal components:
    X_scaled = pipe.steps[0][1].fit_transform(X)  # scaled data
    X_trans = pipe.steps[1][1].fit_transform(X_scaled)  # Dimensionsreduzierung auf die Hauptkomponenten
    y_pred = pipe.predict(X)  # Vorhersagewerte der Pipeline ...

    # 5.2. Plot 3D scatter diagram:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_trans[:, 0], X_trans[:, 1], y, marker="o", c='red', label='actual values')
    ax.set_xlabel("PC 1"), ax.set_ylabel("PC 2"), ax.set_zlabel(target)
    ax.view_init(azim=-60, elev=20)  # position of camera
    plt.tight_layout()
    plt.show()

    # 5.3. Plot regression plane witht min/max of the transformed data:
    x0 = np.linspace(X_trans[:, 0].min(), X_trans[:, 0].max(), num=2)
    x1 = np.linspace(X_trans[:, 1].min(), X_trans[:, 1].max(), num=2)
    xx0, xx1 = np.meshgrid(x0, x1)  # 2x2 - Gitter
    X0, X1 = xx0.ravel(), xx1.ravel()
    yy = pipe.steps[2][1].predict(np.c_[X0, X1]).ravel()  # Prediction values in the regression plane
    ax.plot_trisurf(X0, X1, yy, linewidth=0, alpha=0.3)
    plt.tight_layout()
    plt.show()

