import cupy as np
import time
from solutions_evaluator import epsilon_value
from utils import try_reading_matrix_numpy
import pandas as pd
import sys

EXPERIMENTS_RESULTS_PATH = "../experiments_results_cp/"

SIGNIFICANCE_CONST = 10**(-50)

def MWU_game_algorithm_experiment(payoff_mat, phi=1/2, steps_number=10000):
    payoff_mat = np.array(payoff_mat)
    rows_number = payoff_mat.shape[0]
    cols_number = payoff_mat.shape[1]
    p_0 = np.ones((1, rows_number))
    p_0 = p_0/rows_number
    p_t = p_0
    j_sumed = np.zeros((cols_number, 1))
    smallest_column_payoff = 1
    p_best = p_0
    p_t_sum = np.zeros((1, rows_number))

    start = time.time()
    row_row = []
    col_col = []
    row_col = []
    times = []
    curr_index = 125

    for i in range (1, steps_number + 1):
        payoffs = np.matmul(p_t, payoff_mat)
        j_best_response = np.argmax(payoffs)
        if(payoffs[0, j_best_response] < smallest_column_payoff):
            smallest_column_payoff = payoffs[0, j_best_response]
            p_best = p_t
        j_sumed[j_best_response] += 1
        m_t = payoff_mat[:,j_best_response]
        m_t_negative = (m_t < 0)
        p_t_significant = (p_t > SIGNIFICANCE_CONST)
        to_update = np.logical_or(m_t_negative, p_t_significant[0])
        m_t_updating = np.where(to_update,m_t,0)
        p_t_updating = np.where(to_update,p_t,0)
        p_t = np.multiply((1 - phi * m_t_updating), p_t_updating)
        p_t = p_t/p_t.sum()
        p_t_sum = p_t_sum + p_t
        if(i == curr_index):
            j_distribution = j_sumed / j_sumed.sum()
            print(i)
            now = time.time()
            times.append(now - start)
            row_row.append(max(epsilon_value(p_best, np.transpose(p_best), payoff_mat)))
            col_col.append(max(epsilon_value(np.transpose(j_distribution), j_distribution, payoff_mat)))
            row_col.append(max(epsilon_value(p_best, j_distribution, payoff_mat)))
            start -= (time.time() - now)
            curr_index *= 2
    return times, row_row, col_col, row_col

def run_experiment(A, B, n, phis_bound=7, max_steps_power_mult=7):
    max_steps = 125 * 2 ** max_steps_power_mult
    matrix = -try_reading_matrix_numpy(A,B,n)

    times = []
    row_row = []
    col_col = []
    row_col = []

    phis = [(1/2)**i for i in range(5, phis_bound+1)]
    columns_names = [125 * 2**i for i in range(max_steps_power_mult + 1)]
    for phi in phis:
        result = MWU_game_algorithm_experiment(matrix, phi, max_steps)
        times.append(result[0])
        row_row.append(result[1])
        col_col.append(result[2])
        row_col.append(result[3])
    save_experiments_results(pd.DataFrame(np.array(times), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/times/")
    save_experiments_results(pd.DataFrame(np.array(row_row), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/row_row/")
    save_experiments_results(pd.DataFrame(np.array(col_col), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/col_col/")
    save_experiments_results(pd.DataFrame(np.array(row_col), index=phis, columns=columns_names), gen_name(A, B, n), EXPERIMENTS_RESULTS_PATH + "/row_col/")

def save_experiments_results(data, name, folder=EXPERIMENTS_RESULTS_PATH):
    data.to_csv(folder + name)

def gen_name(A, B, n):
    return "(" + str(A) + "," + str(B) + "," + str(n) + ").csv"

if __name__ == "__main__":
    args = sys.argv
    A, n = int(args[1]), int(args[2])
    if(len(args) > 3):
        phi = int(args[3])
        if(len(args) > 4):
            mult = int(args[4])
            run_experiment(A, A, n, phi, mult)
        else:
            run_experiment(A, A, n, phi)
    else:
        run_experiment(A, A, n)