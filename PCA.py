def PrincipleComponentAnalysis():
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    # Read in the data

    names = ["Date", "Sunshine", "Outdoor", "yield", "pump", "Valve", "consumption"]

    path = "/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv"
    df = pd.read_csv(path, delimiter='\t', names=names, skiprows=0, header=0)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(['Date'], axis=1, inplace=False)


    #separate the data y being consumption and x being the rest of the data
    y = df['consumption']
    x = df.drop(['consumption'], axis=1, inplace=False)

    print(
        x.head(5)
    )

    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x)

    explained_variance_ratio = pca.explained_variance_ratio_
    print(
        explained_variance_ratio
    )


    # Now we can use the linear regression model to predict the consumption
    essential_variables = pd.DataFrame(pca.components_[:2], columns=x.columns)
    print(
        essential_variables
    )

    # Now we can use the linear regression model to predict the consumption
    X_selected = x_pca[:, :2]  # Select the first two principal components
    regressor = LinearRegression()
    regressor.fit(X_selected, y)

    print(
        regressor.score(X_selected, y)
    )
