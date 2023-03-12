"""

This is a python script for the grid search (GS) part of the project.
You can run the python file and input or choose the given options.
You need to specify (choose) the number of grid points, the grid type, the scale type, and the method of processing.
The outputs are the best performance and the corresponding C and gamma, 
computational time and a 3D plot of the five-fold cross-validation accuracy (performance).

Based on our experiment and the result of the grid search, we choose the following parameters:
number of grid points for each parameter: 100 (total 100x100 grid points) (input 100)
grid type: nonuniform (option 2)
scale type: standard (option 1)
method of processing: multiprocessing (option 1)
On our computer, the computational time is about 3.5 minutes.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from multiprocessing import Pool

# function to create a nonuniform grid
def ununiform_grid(boundary, m, n=3):
    """
    Parameters
    ----------
    boundary: boundary of the grid ex: log Gamma ~ [-10,0]
    m: integer. the total number of grid points along one side
    n: integer. grid hierarchy

    Returns
    -------
    grid: numpy array
    """
    sumn = sum(range(1, n + 1))
    x = boundary[0]
    grid = np.array([])
    interval = boundary[1] - boundary[0]

    for i in range(n):
        n_grid = int(m * (n - i) / sumn)

        if i == 0:
            grid = np.append(grid, np.logspace(x, x + interval / n, n_grid, base=10))
        else:
            grid = np.append(
                grid, np.logspace(x, x + interval / n, n_grid, base=10)[1:]
            )

        x += interval / n

    return grid


# calculate the performance of SVC with RBF kernel and given C and gamma using 5-fold cross validation
def svc_cv_performance(input):
    c, g, data_scaled, target = input
    svc = SVC(kernel="rbf", C=c, gamma=g)
    return (c, g, cross_val_score(svc, data_scaled, target, cv=5).mean())


# function to obtain the performance all values of given C and gamma using multiprocessing (20 pools)
def process(inputs):
    print("start processing")
    with Pool(20) as p:
        result = tqdm.tqdm(p.imap(svc_cv_performance, inputs))
        cv_performance_pool = np.array(list(result))

    # print the best performance and the corresponding C and gamma
    max_performance = np.max(cv_performance_pool[:, 2])
    argmax_performance = np.argmax(cv_performance_pool[:, 2])
    print(
        "The best performance is {} with C = {} and gamma = {}".format(
            max_performance,
            cv_performance_pool[argmax_performance, 0],
            cv_performance_pool[argmax_performance, 1],
        )
    )
    return cv_performance_pool


# function to obtain the performance all values of given C and gamma without multiprocessing
def process_without_pool(inputs):
    print("start processing")
    cv_performance = []
    for input in tqdm.tqdm(inputs):
        cv_performance.append(svc_cv_performance(input))

    max_performance = max(cv_performance, key=lambda x: x[2])
    argmax_performance = np.argmax(cv_performance, axis=0)
    print("Best performance: ", max_performance[2])
    print("Best C: ", max_performance[0])
    print("Best gamma: ", max_performance[1])

    return np.array(cv_performance)


# plot the 3D graph of the performance described by C and gamma
def print3D(cv_performance, C_grid, gamma_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # use ax.plot_surface to plot the surface
    sizeX = len(C_grid)
    sizeY = len(gamma_grid)
    ax.plot_surface(
        cv_performance[:, 0].reshape(sizeX, sizeY),
        cv_performance[:, 1].reshape(sizeX, sizeY),
        cv_performance[:, 2].reshape(sizeX, sizeY),
        cmap="viridis",
        linewidth=0.5,
    )
    ax.set_xlabel("C")
    ax.set_ylabel("gamma")
    ax.set_zlabel("performance")
    plt.show()


# function to prepare the input
def prepare_input(C_grid, gamma_grid, scale):
    # load data from heart.csv
    data = pd.read_csv("heart.csv", header=None)

    # make the last column as the target
    target = data[13]
    data = data.drop(13, axis=1)

    if scale == 1:
        # scaled the data using mean = 0 std = 1 using StandardScaler
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        # scaled the data using min = 0 max = 1 using MinMaxScaler
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

    # make data_scaled as a dataframe
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.head()

    inputs = []
    for c in C_grid:
        for g in gamma_grid:
            inputs.append((c, g, data_scaled, target))

    return inputs


if __name__ == "__main__":
    print("Input the number of grid points for each hyperparameter: ")
    n = int(input("Number of grid points = "))
    print("choose the grid type: \n1. uniform grid \n2. nonuniform grid")
    grid_type = int(input("Enter your choice: "))

    if grid_type == 1:
        print("uniform grid")
        # Make an array C: an array from 10^0 to 10^9 with equal spacing with n points
        C_grid = np.linspace(1, 1e9, n)
        # Make an array gamma from 10^-10 to 10^0
        gamma_grid = np.linspace(1e-10, 1, n)
    elif grid_type == 2:
        print("nonuniform grid")
        C_grid = ununiform_grid(np.array([0, 9]), n)
        gamma_grid = ununiform_grid(np.array([-10, 0]), n)

    print("Choose the scale: \n1. StandardScaler \n2. MinMaxScaler")
    scale = int(input("Enter your choice: "))

    inputs = prepare_input(C_grid, gamma_grid, scale)

    print("Choose the method of processing: ")
    print(
        "1. Multiprocessing (GS with pool) \n2. Traditional processing (GS without pool)"
    )
    choice = int(input("Enter your choice: "))
    if choice == 1:
        # start timing
        start = time.time()
        cv_performance = process(inputs)
        # end timing
        end = time.time()
        print("Time: ", end - start)
    elif choice == 2:
        # start timing
        start = time.time()
        cv_performance = process_without_pool(inputs)
        # end timing
        end = time.time()
        print("Time: ", end - start)
    print3D(cv_performance, C_grid, gamma_grid)
