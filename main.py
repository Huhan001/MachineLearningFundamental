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

def chaprterTwo():
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import tarfile
    import urllib.request
    import matplotlib.pyplot as plt

    # ⚠️ fetch the data.
    def load_housing_data():
        tarbal_path = Path("datasets/housing.tgz")
        if not tarbal_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarbal_path)
            with tarfile.open(tarbal_path) as housing_tarbal:
                housing_tarbal.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))

    housing = load_housing_data()
    # print(housing.info())
    # a way to know the data within.
    # print(housing["ocean_proximity"].value_counts(sort=True))
    # print(housing.describe())
    # housing.hist(bins=50, figsize=(12,8))
    # plt.show()

    # setting data aside

    #def shuffle_and_split_data(data, test_ration):
     #   shuffled_indeces = np.random.permutation(len(data))
     #   test_set = int(len(data) * test_ration)
      #  test_indices = shuffled_indeces[:test_set]
      #  train_indices = shuffled_indeces[test_set:]
       # return data.iloc[train_indices], data.iloc[test_indices]

    #train_set, test_set = shuffle_and_split_data(housing, 0.2)
    #print(len(train_set), len(test_set))

    #from zlib import crc32
    #def is_id_in_test_set(identifier, test_ratio):
       # return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

   #def split_data_with_id_hash(data, test_ratio, id_column):
        #ids = data[id_column]
        #in_test_set = ids.apply(lambda id_:is_id_in_test_set(id_, test_ratio))
        #return data.loc[~in_test_set], data.loc[in_test_set]

    #housing_with_id = housing.reset_index()
    #train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

    #housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    #train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(housing, test_size= 0.2, random_state= 42)



if __name__ == "__main__":
    chaprterTwo()
