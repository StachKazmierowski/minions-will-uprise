import numpy as np
import pandas as pd
MATRICES_PATH = "../payoff_matrices/"
MATRICES_RAPORTS_PATH = "../matrices_creation_reports/"
RESULT_PRECISION = 10**(-20)
PHI = 1/8
STEPS_NUMBER = 10000
RESULTS_PATH = "../results/"

def next_divide(divide):
    n = divide.shape[1]
    div_num = divide.shape[0]
    dev_tmp = np.empty((0, n), int)
    for i in range(div_num):
        tmp = divide[i][:]
        for j in range(n):
            if (j == 0 or tmp[j] < tmp[j - 1]):
                tmp[j] = tmp[j] + 1
                dev_tmp = np.append(dev_tmp, tmp.reshape(1, n), axis=0)
                tmp[j] = tmp[j] - 1
    return (np.unique(dev_tmp, axis=0))

def divides(A, n):
    if (A == 0):
        return np.zeros((1, n))
    devs = np.zeros((1, n))
    devs[0][0] = 1
    for i in range(A - 1):
        devs_next = next_divide(devs)
        devs = devs_next
    return (devs)

def try_reading_matrix(A, B, fields_number):
    try:
        path = MATRICES_PATH + str(fields_number) + "_fields/payoff_matrix(" + str(A) + \
           "," + str(B) + "," + str(fields_number) + ").csv"
        payoff_mat = pd.read_csv(path, index_col=0)
    except:
        print("Loaded failed")
        print(A, B, fields_number)
        # return
    return payoff_mat

def try_reading_matrix_numpy(A ,B, fields_number):
    payoff_mat = try_reading_matrix(A, B, fields_number).to_numpy()
    return payoff_mat
