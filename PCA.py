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
    #rotation_animation.save('rotation_animation.gif', writer='imagemagick')  # Save the animation as a GIF file
    plt.show()


