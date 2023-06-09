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

    # stratified sampling
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins= [0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels = [1, 2, 3, 4, 5])
    # lets plot this
    #housing["income_cat"].value_counts().sort_index().plot(kind = "bar")
    #plt.xlabel("income category")
    #plt.ylabel("count")
    #plt.show()

    #print(housing["income_cat"].value_counts() / len(housing)) #percentage of each category.

    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits= 10, test_size= 0.2, random_state= 42)
    strat_splits = []
    for train_index, test_index in splitter.split(housing, housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n, strat_test_set_n])

    #now lets use the first split.
    #strat_train_set, strat_test_set = strat_splits[0]
    # there is however a much shorter way to get stratified data with scipy

    strat_train_set, strat_test_set = train_test_split(housing, test_size= 0.2,
                                                       random_state = 42,
                                                       stratify = housing["income_cat"])

    # after the stratfied. we wont use the income again so we shall drop it.
    for set_ in (strat_train_set, strat_test_set):
        set_.drop(columns = "income_cat", axis = 1, inplace = True)

    # visualizeing the data
    housing_data = strat_train_set.copy() # we copy to avoid mishandlind the data.
    housing_data.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.2,
                      grid = True)
    #plt.show()

    # we can do better with the visualizarion, by adding color
    housing_data.plot(kind = "scatter", x = "longitude", y = "latitude",
                      s = housing_data["population"] / 100, label = "population",
                      c = "median_house_value", cmap = "jet", colorbar = True,
                      legend = True, sharex = False, figsize = (12, 8))
    #plt.show()

    # since the data is not too large, we can compute the corelation matrix
    # to see how well corelated they are to the house prices.
    corr_matrix = housing_data.drop(columns = "ocean_proximity").corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))

    #creating new attributes
    housing_data["rooms_per_house"] = housing_data["total_rooms"] / housing_data["households"]
    housing_data["bedrooms_ratio"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
    housing_data["people_per_house"] = housing_data["population"] / housing_data["households"]
    #print(housing_data.drop(columns = "ocean_proximity").corr()["median_house_value"].sort_values())

    # fetch the data now preparing for machine learning
    houses_train = housing_data.drop(columns = "median_house_value", axis = 1)
    houses_labels = housing_data["median_house_value"].copy()

    # Get rid of the corresponding districts. 1
    # Get rid of the whole attribute. 2
    # Set the missing values to some value (zero, the mean, the median, etc.). This is called imputation.

    #print(houses_train["ocean_proximity"].value_counts())
    houses_train.dropna(subset =["total_bedrooms"]) # option 1 🅧
    houses_train.drop("total_bedrooms", axis = 1) # option 2
    median = houses_train["total_bedrooms"].median()
    houses_train["total_bedrooms"].fillna(median) # options 3

    # you can employ the simple imputer by sklearn.✨

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy = "median")

    # to use it, create a copy of all numerical attributes.
    housing_num = houses_train.select_dtypes(include = [np.number])
    imputer.fit(housing_num)
    x = imputer.transform(housing_num)

    #when trying to recover x.
    #housing_tr = pd.DataFrame(x, columns = housing_num.columns,index= housing_num.index)
    #print(housing_tr.head(5))

    # handling text and categorical attributes
    # we can use the ordinal encoder to encode the text to numbers.
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    housing_cat = houses_train[["ocean_proximity"]]
    #houses_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    from sklearn.preprocessing import OneHotEncoder
    cat_onehot = OneHotEncoder()
    housing_onehot = cat_onehot.fit_transform(housing_cat)
    #print(housing_onehot.toarray())

    # Feature Scaling
    # we can use the min-max scaler or the standard scaler.
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_min_maxed = min_max_scaler.fit_transform(x)

    from sklearn.preprocessing import StandardScaler
    standard_scaler = StandardScaler()
    housing_standard = standard_scaler.fit_transform(x)

    # Transformation
    from sklearn.linear_model import LinearRegression
    from sklearn.compose import TransformedTargetRegressor

    newfictionaldata = housing_data[["median_income"]].iloc[:5]

    model = TransformedTargetRegressor(LinearRegression(),
                                       transformer=StandardScaler())
    model.fit(housing_data[["median_income"]], houses_labels)
    prediction = model.predict(newfictionaldata)

    # custom tranformer, your own function tranformer
    from sklearn.preprocessing import FunctionTransformer
    log_tranformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_tranformer.transform(housing_data[["population"]])

    # custome tranformer with base estimator
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted

    class standardScalerClone(BaseEstimator, TransformerMixin):
        def __init__(self, with_mean = True):
            self.with_mean = with_mean

        def fit(self,X, y = None):
            X = check_array(X)
            self.mean_ = np.mean(X, axis = 0)
            self.scale_ = np.std(X, axis = 0)
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            check_is_fitted(self)
            X = check_array(X)
            assert self.n_features_in_ == X.shape[1]
            if self.with_mean:
                X -= self.mean_
            X /= self.scale_
            return X

    # pipeline

    from sklearn.pipeline import Pipeline # fi you want to name the transformers.
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline([("impute", SimpleImputer(strategy = "median")),
                             ("standardize", StandardScaler())])


    # if you do not want to name the transfomers
    from sklearn.pipeline import make_pipeline
    new_num_pipeline = make_pipeline(SimpleImputer(strategy = "median"),
                                     StandardScaler())

    # understanding pipelines
    housing_num_prepared = new_num_pipeline.fit_transform(housing_num)
    #print(housing_num_prepared[:5])

    #if i wanted to recover the names i would use
    df_housing_num_prepared = pd.DataFrame(housing_num_prepared,
                                           columns = new_num_pipeline.get_feature_names_out())
    #print(df_housing_num_prepared)

    # combining the numerical and categorical pipelines
    from sklearn.compose import ColumnTransformer
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    cat_pipeline = make_pipeline(SimpleImputer(strategy = "most_frequent"),
                                 OneHotEncoder(handle_unknown="ignore"))
    porerprocessing = ColumnTransformer([("num", new_num_pipeline, num_attribs),
                                        "cat", cat_pipeline, cat_attribs])

    # use a different one without naming.

    from sklearn.compose import make_column_transformer, make_column_selector
    preprocessing = make_column_transformer(
        (new_num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object))
    )

    housing_prepared =  preprocessing.fit_transform(houses_train)
    pd.DataFrame(housing_prepared, columns= preprocessing.get_feature_names_out(),
                       index= houses_train.index)
    # need to make sure you are sizing up the indexes just right.

    def column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ["ratio"]

    def ratio_pipeline():
        return make_pipeline(
            SimpleImputer(strategy = "median"),
            FunctionTransformer(column_ratio, feature_names_out=ratio_name),
            StandardScaler()
        )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy = "median"),
        FunctionTransformer(np.log, feature_names_out= "one-to-one"),
        StandardScaler()
    )

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy = "median"),
        StandardScaler()
    )

    # the main processor connector.

    Ppreprocessing = make_column_transformer(
        (ratio_pipeline(), ["total_rooms", "total_bedrooms"]),
        (ratio_pipeline(), ["total_rooms", "households"]),
        (ratio_pipeline(), ["population", "households"]),
        (log_pipeline, ["population", "households",
                        "total_bedrooms","total_rooms", "median_income",]),
        (cat_pipeline, make_column_selector(dtype_include=object)),
        remainder= default_num_pipeline # for remaining columns
    )

    housing_prepared = Ppreprocessing.fit_transform(houses_train)
    #print(housing_prepared.shape)
    #print(pd.DataFrame(housing_prepared, columns= Ppreprocessing.get_feature_names_out(),index= houses_train.index))

    # thats how you create a pipeline. for cleaning the data.

    # after the cleaning, now our data will be smooth sailing for the machine learning
    # algorithms.

    from sklearn.linear_model import LinearRegression
    lin_reg = make_pipeline(Ppreprocessing, LinearRegression())
    lin_reg.fit(houses_train, houses_labels)
    linear_housing_predictions = lin_reg.predict(houses_train)
   # print(linear_housing_predictions[:5].round(-2))# -2 rounds to the nearest 100
   # print(houses_labels[:5].values)

    # for comparison sake to see how it fairs with the rest of the tests.
    # get the measurements via mean_squared_error.
    from sklearn.metrics import mean_squared_error

    lin_rsme = mean_squared_error(houses_labels, linear_housing_predictions, squared=False)
    #print(lin_rsme)
    # the figure is 68893.4044
    # prediction error of $68,893.40 is not satisfying. with median house values ranging
    # from $120,000 to $265,000, this is not a good model.

    #lets use aq better model, the decision tree regressor.
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = make_pipeline(Ppreprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(houses_train, houses_labels)
    tree_housing_predictions = tree_reg.predict(houses_train)
    tree_rsme = mean_squared_error(houses_labels, tree_housing_predictions, squared=False)
    #print(tree_rsme)

    # Better Evaluation Using Cross-Validation
    from sklearn.model_selection import cross_val_score
    # using tree reg from pipeline.
    tree_rsmes = -cross_val_score(tree_reg, houses_train, houses_labels,
                                  scoring="neg_root_mean_squared_error", cv=10)
    # the scoring is very important here
    #print(pd.Series(tree_rsmes).describe())


    # let use the random forest regressor
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = make_pipeline(Ppreprocessing, RandomForestRegressor(random_state=42))
    forest_rsmes = -cross_val_score(forest_reg, houses_train, houses_labels,
                                    scoring="neg_root_mean_squared_error", cv=10)



    from sklearn.model_selection import GridSearchCV
    full_pipeline = Pipeline([
        ("preprocessing", Ppreprocessing),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    param_grid = [
        {"preprocessing_geo_n_clusters": [5, 8, 10],
         "random_forest__max_features": [2, 4, 6, 8]},
        {"preprocessing_geo_n_clusters": [10, 15],
            "random_forest__bootstrap": [6, 8, 10]}
    ]

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5,
                                 scoring="neg_root_mean_squared_error")
    grid_search.fit(houses_train, houses_labels)
    print(grid_search.best_params_)

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_distribs = {"preprocessing_geo_n_clusters": randint(low=5, high=10),
                      "random_forest__max_features": randint(low=2, high=8)}
    rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring="neg_root_mean_squared_error",
                                    random_state=42)
    rnd_search.fit(houses_train, houses_labels)
    print(rnd_search.best_params_)
