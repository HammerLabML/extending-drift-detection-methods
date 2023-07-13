from copy import deepcopy
from os.path import isfile
from sklearn.datasets import fetch_openml, fetch_covtype
import random
import numpy as np
import requests


# store some datasets globally because loading them from file again and again is too slow
try:
    mnist_X = np.load("mnist_X.npy", allow_pickle=True)
    mnist_y = np.load("mnist_y.npy", allow_pickle=True)
except FileNotFoundError:
    print("Downloading mnist dataset ...")
    mnist_X, mnist_y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    np.save("mnist_X.npy", mnist_X)
    np.save("mnist_y.npy", mnist_y)
try:
    covtype_X = np.load("covtype_X.npy")
except FileNotFoundError:
    print("Downloading forest covertype dataset ...")
    covtype_X = fetch_covtype()["data"]
    assert covtype_X.shape[0] == 581012
    assert covtype_X.shape[1] == 54
    np.save("covtype_X.npy", covtype_X)
if not isfile("rialto.npy") or not isfile("rialto_labels.npy"):
    # Download from https://github.com/vlosing/driftDatasets/tree/master/realWorld/rialto
    print("Downloading rialto dataset ...")
    open("rialto.data", "wb").write(requests.get("https://raw.githubusercontent.com/vlosing/driftDatasets/master/realWorld/rialto/rialto.data").content)
    open("rialto.labels", "wb").write(requests.get("https://raw.githubusercontent.com/vlosing/driftDatasets/master/realWorld/rialto/rialto.labels").content)
    X = np.genfromtxt('rialto.data', dtype=float, delimiter=' ')
    y = np.genfromtxt('rialto.labels', dtype=int, delimiter=' ')
    assert X.shape[0] == 82250
    assert X.shape[1] == 27
    assert X.shape[0] == y.shape[0]
    np.save("rialto.npy", X)
    np.save("rialto_labels.npy", y)
if not isfile("music.npy"):
    try:
        full_dataset = np.genfromtxt('music.csv', dtype=float, delimiter=',')[1:, :]  # First line is table header
    except FileNotFoundError:
        print("Downloading music dataset ...")
        open("music.csv", "wb").write(requests.get("https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/music.csv").content)
        full_dataset = np.genfromtxt('music.csv', dtype=float, delimiter=',')[1:, :]  # First line is table header
    for i in range(6, full_dataset.shape[1]):
        data_min = np.nanpercentile(a=full_dataset[:, i], q=0.1)
        data_max = np.nanpercentile(a=full_dataset[:, i], q=99.9)
        if (data_max - data_min) > 0:
            full_dataset[:, i] = np.minimum(1.0, np.maximum(0.0, (full_dataset[:, i] - data_min) * (1.0 / (data_max - data_min))))
    np.save("music.npy", full_dataset)

def mnist_dataset(length, drifts):
    assert mnist_X.shape[1] == 784
    data = np.zeros(shape=(length, 14*14))
    if isinstance(drifts, int):
        if length == 1600 and drifts == 2:
            real_drifts = [random.randint(380, 700), random.randint(900, 1220)]
        else:
            real_drifts = [random.randint(int(((i+1)/(drifts+1)-0.1)*length), int(((i+1)/(drifts+1)+0.1)*length)) for i in range(drifts)]
    elif isinstance(drifts, list):
        real_drifts = drifts
    else:
        exit()

    concept = random.randint(0, 4)
    desired_label = str(random.randint(0, 9))
    for i in range(length):
        if i in real_drifts:
            concept = (concept + 1) % 5
        j = random.randint(0, mnist_X.shape[0]-1)
        while mnist_y[j] != desired_label:
            j = (j+1) % mnist_X.shape[0]
        digit = deepcopy(mnist_X[j, :])
        digit = digit.reshape(28, 28)
        if concept == 0:
            pass
        elif concept == 1:
            digit[0:27, :] = digit[1:28, :]
            digit[27, :] = 0
        elif concept == 2:
            digit[:, 0:27] = digit[:, 1:28]
            digit[:, 27] = 0
        elif concept == 3:
            digit[1:28, :] = digit[0:27, :]
            digit[0, :] = 0
        elif concept == 4:
            digit[:, 1:28] = digit[:, 0:27]
            digit[:, 0] = 0
        else:
            print("ERROR: bad concept")
            exit()
        data[i, :] = digit[0:28:2, 0:28:2].reshape(14*14)

    data *= (1.0/255.0)
    assert np.min(data) == 0.0
    assert np.max(data) == 1.0
    return data, np.array(real_drifts)-0.5


def covtype_dataset(length, drifts):
    assert covtype_X.shape[1] == 54
    data = np.zeros((length, 54))
    if isinstance(drifts, int):
        if length == 1600 and drifts == 2:
            real_drifts = [random.randint(380, 700), random.randint(900, 1220)]
        else:
            real_drifts = [random.randint(int(((i + 1) / (drifts + 1) - 0.1) * length), int(((i + 1) / (drifts + 1) + 0.1) * length)) for i in range(drifts)]
    elif isinstance(drifts, list):
        real_drifts = drifts
    else:
        exit()
    X_std = np.array([279.9844933, 111.91362469, 7.48823537, 212.54917268, 58.29518146, 1559.25352805, 26.76986577, 19.76868014, 38.27449629, 1324.19407022])
    assert X_std.shape[0] == 10
    shift_mean = []
    invert_feature = []
    for i in range(length):
        if i in real_drifts:
            new_shift_mean = random.randint(0, 9)
            while len(shift_mean) > 0 and new_shift_mean in shift_mean:
                new_shift_mean = random.randint(0, 9)
            shift_mean.append(new_shift_mean)
            new_invert_feature = random.randint(10, 53)
            while len(invert_feature) > 0 and new_invert_feature in invert_feature:
                new_invert_feature = random.randint(10, 53)
            invert_feature.append(new_invert_feature)
        randnr = random.randint(0, covtype_X.shape[0]-1)
        data[i, :] = covtype_X[randnr, :]
        for j in shift_mean:
            data[i, j] += 1.0*X_std[j]
        for j in invert_feature:
            data[i, j] = 1 - data[i, j]
    for i in range(data.shape[1]):
        data_min = np.nanpercentile(a=data[:, i], q=0.1)
        data_max = np.nanpercentile(a=data[:, i], q=99.9)
        if (data_max-data_min) > 0:
            data[:, i] = np.minimum(1.0, np.maximum(0.0, (data[:, i] - data_min) * (1.0/(data_max-data_min))))
    return data, np.array(real_drifts)-0.5


def rialto_dataset(length, drifts):
    X = np.load("rialto.npy")
    y = np.load("rialto_labels.npy")
    data = np.zeros((length, X.shape[1]))
    if isinstance(drifts, int):
        if length == 1600 and drifts == 2:
            real_drifts = [random.randint(380, 700), random.randint(900, 1220)]
        else:
            real_drifts = [random.randint(int(((i + 1) / (drifts + 1) - 0.1) * length), int(((i + 1) / (drifts + 1) + 0.1) * length)) for i in range(drifts)]
    elif isinstance(drifts, list):
        real_drifts = drifts
    else:
        exit()
    concept = random.randint(0, 9)
    for i in range(length):
        if i in real_drifts:
            new_concept = random.randint(0, 9)
            while concept == new_concept:
                new_concept = random.randint(0, 9)
            concept = new_concept
        j = random.randint(0, X.shape[0]-1)
        while y[j] != concept:
            j = (j+1) % X.shape[0]
        data[i, :] = X[j, :]
    return data, np.array(real_drifts)-0.5


def music_dataset(length: int = 1600, drifts: int = 2) -> tuple[np.ndarray, list]:
    assert drifts == 2
    full_dataset = np.load("music.npy")
    X = full_dataset[:, 6:]
    y = full_dataset[:, :6]
    assert X.shape[1] == 72
    data = np.zeros((length, 72))
    if isinstance(drifts, int):
        if length == 1600 and drifts == 2:
            real_drifts = [random.randint(380, 700), random.randint(900, 1220)]
        else:
            real_drifts = [random.randint(int(((i + 1) / (drifts + 1) - 0.1) * length), int(((i + 1) / (drifts + 1) + 0.1) * length)) for i in range(drifts)]
    elif isinstance(drifts, list):
        real_drifts = drifts
    else:
        exit()
    concept = 0
    for i in range(length):
        if i in real_drifts:  # Possible alternative: 1-4-5
            if concept == 0:
                concept = 3
            elif concept == 3:
                concept = 5
            else:
                print("Error")
                return
        j = random.randint(0, X.shape[0] - 1)
        if random.random() > 0.333:
            while y[j, concept] != 1:
                j = (j + 1) % X.shape[0]
            data[i, :] = X[j, :]
        else:
            while y[j, 0] != 0 and y[j, 3] != 0 and y[j, 5] != 0:
                j = (j + 1) % X.shape[0]
            data[i, :] = X[j, :]
    return data, np.array(real_drifts)-0.5


def trivial_dataset(length=1600, drifts=2):
    data = np.zeros((length, 100))
    if isinstance(drifts, int):
        if length == 1600 and drifts == 2:
            real_drifts = [random.randint(380, 700), random.randint(900, 1220)]
        else:
            real_drifts = [random.randint(int(((i + 1) / (drifts + 1) - 0.1) * length), int(((i + 1) / (drifts + 1) + 0.1) * length)) for i in range(drifts)]
    elif isinstance(drifts, list):
        real_drifts = drifts
    else:
        exit()
    for i in range(1, length):
        if i in real_drifts:
            if data[i-1, 0] > 0.5:
                data[i, :] = 0.1*random.random()
            else:
                data[i, :] = 1.0 - 0.1*random.random()
        else:
            if data[i-1, 0] > 0.5:
                data[i, :] = 1.0 - 0.1*random.random()
            else:
                data[i, :] = 0.1*random.random()
    return data, np.array(real_drifts)-0.5
