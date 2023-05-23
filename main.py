#------------------------------------------------------------ Chapter 1.
def checkyourself():
    # Example 1- 1 Training and running a linear model using Scikit - learn.
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression


    # ⚠️ ( data input )
    data_root = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
    lifesat = pd.read_csv(data_root)
    x = lifesat[["GDP per capita (USD)"]].values
    y = lifesat[["Life satisfaction"]].values

    # 🥊( visualization )
    lifesat.plot(kind="scatter", grid=True, x="GDP per capita (USD)",
                 y="Life satisfaction")
    plt.axis([23_500, 62_500, 4, 9])
    plt.show()

    # 👀 ( select linear model )
    model = LinearRegression()

    # 📌( train the model )
    model.fit(x, y)

    x_new = [[37_655.2]]
    print(model.predict(x_new))


def usingKneares():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=3)

    #⚠️
    data_root = "https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
    lifesat = pd.read_csv(data_root)
    x = lifesat[["GDP per capita (USD)"]].values
    y = lifesat[["Life satisfaction"]].values

    # 📌
    model.fit(x, y)
    x_new = [[37_655.2]]
    print(model.predict(x_new))

    #🥊
    a = model.kneighbors_graph(x)
    print(a.toarray())


if __name__ == "__main__":
    #checkyourself()
    usingKneares()
