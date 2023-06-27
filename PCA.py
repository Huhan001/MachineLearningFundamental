def PrincipleComponentAnalysis():
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Read in the data

    names = ["Date", "Sunshine", "Outdoor", "yield", "pump", "Valve", "consumption"]

    #Used for simplicity of the code
    predictors = ["Sunshine", "Outdoor", "yield", "pump", "Valve"]
    target = "consumption"

    path = "/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv"
    df = pd.read_csv(path, delimiter='\t', header=0, names=names)
    df = df.drop(columns=["Date"])

    #separate the data y being consumption and x being the rest of the data
    x = np.c_[df[predictors]]
    y = np.c_[df[target]]

    #sepate the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    #create the pipeline
    pipe = make_pipeline(StandardScaler(), PCA(2), LinearRegression())
    pipe.fit(x_train, y_train)

    #calculate the score of the model
    test_percent = pipe.score(x_test, y_test) * 100
    train_percent = pipe.score(x_train, y_train) * 100

    #print the score of the model
    print("The training scor is: {}%".format(train_percent))
    print("The testing score is: {}%".format(test_percent))

    #plot the data
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x_scaled = pipe.steps[0][1].fit_transform(x)
    x_trans = pipe.steps[1][1].fit_transform(x_scaled)
    y_pred = pipe.predict(x)

    def update_rotation(num, ax, fig):
        ax.view_init(elev=num, azim=num)  # Update the view angles for each frame

    import matplotlib.animation as animation
    plt.switch_backend('TkAgg')
    # Create the animation
        # Plot the data
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("PC1"), ax.set_ylabel("PC2"), ax.set_zlabel("Consumption")
    ax.scatter(x_trans[:, 0], x_trans[:, 1], y_pred, c='r', label='Predicted')
    ax.scatter(x_trans[:, 0], x_trans[:, 1], y.ravel(), c='b', label='Actual')
    ax.legend()
    plt.tight_layout()

        # Create the animation
    rotation_animation = animation.FuncAnimation(fig, update_rotation, frames=360, fargs=(ax, fig), interval=50)
    rotation_animation.save('rotation_animation.gif', writer='imagemagick')  # Save the animation as a GIF file
    plt.show()




def omptimalComponents():
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Read in the data
    names = ["Date", "Sunshine", "Outdoor", "yield", "pump", "Valve", "consumption"]

    # Used for simplicity of the code
    predictors = ["Sunshine", "Outdoor", "yield", "pump", "Valve"]
    target = "consumption"

    path = "/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv"
    df = pd.read_csv(path, delimiter='\t', header=0, names=names)
    df = df.drop(columns=["Date"])

    # Separate the data: y is the consumption and x is the rest of the data
    x = np.c_[df[predictors]]
    y = np.c_[df[target]]

    # Separate the data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    #how many principal components of the other variables x is best to be considered?
    #create the pipeline
    pipe = make_pipeline(StandardScaler(), PCA(), LinearRegression())

    #create a list of the number of components to consider
    components = [1, 2, 3, 4, 5]
    #create a list of the scores
    scores = []
    #loop through the components
    for component in components:
        #set the components in the pipeline
        pipe.steps[1][1].n_components = component
        #fit the model
        pipe.fit(x_train, y_train)
        #calculate the score
        score = pipe.score(x_test, y_test)
        #append the score to the scores list
        scores.append(score)

    #plot the scores
    import matplotlib.pyplot as plt
    plt.plot(components, scores)
    plt.xlabel("Number of components")
    plt.ylabel("Score")
    plt.show()

    #print the scores
    print("The scores are: {}".format(scores))
    print("The best score is: {}".format(max(scores)))
    print("The best number of components is: {}".format(components[scores.index(max(scores))]))


def homework():
    import numpy as np
    import pandas as pd



    # Read in the data
    path = "/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv"
    data = pd.read_csv(path, delimiter='\t')

    #separate the data
    # Split the data into x and y
    x = data[
        ['Sunshine duration [h/day]', 'Outdoor temperature [°C]', 'Solar yield [kWh/day]', 'Solar pump [h/day]',
         'Valve [h/day]']]
    y = data['Gas consumption [kWh/day]']


    #perform standardization
    from sklearn.preprocessing import StandardScaler

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Standardize the predictor values (x)
    x_standardized = scaler.fit_transform(x)

    # Create a new DataFrame with the standardized predictor values
    x_standardized_df = pd.DataFrame(x_standardized, columns=x.columns)


    # calculate th covariance matrix
    cov_matrix = x_standardized_df.cov()

    # Display the covariance matrix
    print(cov_matrix)


    # compute the eigenvalues and eigenvectors
    import numpy as np

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    eigenvalue_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_indices]
    eigenvectors = eigenvectors[:, eigenvalue_indices]

    # Display the eigenvalues and eigenvectors
    print("Eigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)

    import numpy as np

    # Calculate the covariance matrix
    covariance_matrix = np.cov(x_standardized.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    eigenvalue_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[eigenvalue_indices]
    sorted_eigenvectors = eigenvectors[:, eigenvalue_indices]

    # Rank the eigenvectors based on eigenvalues
    ranked_eigenvectors = sorted_eigenvectors

    # Print the ranked eigenvectors
    for i, eigenvector in enumerate(ranked_eigenvectors):
        print("Principal Component", i + 1)
        print(eigenvector)
        print()


    import  matplotlib.pyplot as plt
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, _ = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Calculate the explained variance ratio
    explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    # Calculate the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(sorted_eigenvalues) + 1), explained_variance_ratio, marker='o', linestyle='-', color='b')
    plt.plot(np.arange(1, len(sorted_eigenvalues) + 1), cumulative_variance, marker='o', linestyle='--', color='r')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.legend(['Explained Variance Ratio', 'Cumulative Explained Variance'])
    plt.grid(True)
    plt.show()

def heatmap():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read in the data
    path = "/Users/humphreyhanson/Projects/MachineLearning/datasets/housing/Heating-data.csv"
    data = pd.read_csv(path, delimiter='\t')

    #separate the data
    # Split the data into x and y
    x = data[
        ['Sunshine duration [h/day]', 'Outdoor temperature [°C]', 'Solar yield [kWh/day]', 'Solar pump [h/day]',
         'Valve [h/day]']]
    y = data['Gas consumption [kWh/day]']

    #perform standardization
    from sklearn.preprocessing import StandardScaler
    model = StandardScaler()
    x_standardized = model.fit_transform(x)

    from sklearn.decomposition import PCA
    model2 = PCA(2)
    x_pca = model2.fit_transform(x_standardized)

    #plot the heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.heatmap(model2.components_, cmap='plasma')
    plt.yticks([0, 1], ['PC 1', 'PC 2'], rotation=360, va='center')
    plt.xticks(range(len(x.columns)), x.columns, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal Components")
    plt.title("Heatmap of Principal Components")
    plt.show()

    #plot the scatterplot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='plasma')
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Scatterplot of Principal Components")
    plt.show()

    # plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(model2.explained_variance_ratio_) + 1), model2.explained_variance_ratio_, marker='o',
             linestyle='-', color='b')
    plt.plot(np.arange(1, len(model2.explained_variance_ratio_) + 1), np.cumsum(model2.explained_variance_ratio_),
             marker='o', linestyle='--', color='r')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.legend(['Explained Variance Ratio', 'Cumulative Explained Variance'])
    plt.grid(True)
    plt.show()

