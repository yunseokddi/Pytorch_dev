import matplotlib.pyplot as plt
import pandas as pd


def plot(x_data, y_data, classes=range(5), plots_per_class=10):
    f, ax = plt.subplots(5, sharex=True, sharey=True, figsize=(10, 10))
    for i in classes:
        for j in range(plots_per_class):
            ax[i].set_title("class{}".format(i))
            ax[i].plot(x_data[y_data == i].iloc[j, :], color="blue", alpha=.5)
    plt.show()

def class_spec(data, classnumber, n_samples, y_train):
    fig = plt.figure(figsize=(10, 13))
    if type(data) == pd.DataFrame:
        plt.imshow(data[y_train == classnumber].iloc[:n_samples, :],
                   cmap="viridis", interpolation="nearest")
    else:
        plt.imshow(data[y_train == classnumber][:n_samples, :],
                   cmap="viridis", interpolation="nearest")
    plt.title("class{}".format(classnumber))
    plt.show()
