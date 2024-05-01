import numpy as np
from numpy.random import multivariate_normal
from scipy.stats import wishart
import matplotlib.pyplot as plt

epsilon =1e-5

def Normal_Wishart(mu_0, lamb, W, nu, seed=None):
    Lambda = wishart(df=nu, scale=W, seed=seed).rvs()
    cov = np.linalg.inv(lamb * Lambda)
    mu = multivariate_normal(mu_0, cov)
    return mu, Lambda, cov

def ranking_matrix(N, M, filename, sep=","):
    R = np.zeros((N, M), dtype=float)
    movieID = {}
    nextID = 0
    f = open(filename, "r")
    for line in f:
        if line[0] == 'u':
            # this is a comment
            continue
        user, movie, ranking, timestamp = line.split(sep)
        if movie not in movieID:
            movieID[movie] = nextID
            nextID += 1
        R[int(user) - 1, int(movieID[movie])] = float(ranking)
    return R, movieID


def BPMF(R, R_test, U_in, V_in, T, D, threshold, lowest_rating, highest_rating,
         mu_0=None, Beta_0=None, W_0=None, nu_0=None):

    def ranked(i, j):  # function telling if user i ranked movie j in the train dataset.
        if R[i, j] != 0:
            return True
        else:
            return False

    def ranked_test(i, j):
        if R_test[i, j] != 0:
            return True
        else:
            return False

    N = R.shape[0]
    M = R.shape[1]

    R_predict = np.zeros((N, M))
    U_old = np.array(U_in)  # initial_value
    V_old = np.array(V_in)  # initial_value

    train_err_list = []
    test_err_list = []
    train_epoch_list = []

    alpha = 2
    mu_u = np.zeros((D, 1))
    mu_v = np.zeros((D, 1))
    Lambda_U = np.eye(D)
    Lambda_V = np.eye(D)


    pairs_test = 0
    pairs_train = 0
    for i in range(N):
        for j in range(M):
            if ranked(i, j):
                pairs_train = pairs_train + 1
            if ranked_test(i, j):
                pairs_test = pairs_test + 1


    if mu_0 is None:
        mu_0 = np.zeros(D)
    if nu_0 is None:
        nu_0 = D
    if Beta_0 is None:
        Beta_0 = 2
    if W_0 is None:
        W_0 = np.eye(D)

    for t in range(T):
        Beta_0_star = Beta_0 + N
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)

        V_average = np.sum(V_old, axis=1) / N
        S_bar_V = np.dot(V_old, np.transpose(V_old)) / N
        mu_0_star_V = (Beta_0 * mu_0 + N * V_average) / (Beta_0 + N)
        W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * np.dot(
            np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
        W_0_star_V = np.linalg.inv(W_0_star_V_inv)
        mu_V, Lambda_V, cov_V = Normal_Wishart(mu_0_star_V, Beta_0_star, W_0_star_V, nu_0_star, seed=None)


        U_average = np.sum(U_old, axis=1) / N
        S_bar_U = np.dot(U_old, np.transpose(U_old)) / N
        mu_0_star_U = (Beta_0 * mu_0 + N * U_average) / (Beta_0 + N)
        W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * np.dot(
            np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
        W_0_star_U = np.linalg.inv(W_0_star_U_inv)
        mu_U, Lambda_U, cov_U = Normal_Wishart(mu_0_star_U, Beta_0_star, W_0_star_U, nu_0_star, seed=None)


        U_new = np.array([])
        V_new = np.array([])

#smaple U
        for i in range(N):
            Lambda_U_2 = np.zeros((D, D))
            mu_i_star_1 = np.zeros(D)
            for j in range(M):
                if ranked(i, j):
                    Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
                                                     np.array((V_old[:, j]), ndmin=2))
                    mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1

            Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
            Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U+ epsilon * np.eye(Lambda_i_star_U.shape[0]))

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U,mu_U)
            mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)


            U_new = np.append(U_new, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))


        U_new = np.transpose(np.reshape(U_new, (N, D)))

#sample V
        for j in range(M):
            Lambda_V_2 = np.zeros((D, D))
            mu_i_star_1 = np.zeros(D)
            for i in range(N):
                if ranked(i, j):
                    Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(U_new[:, i], ndmin=2)),
                                                     np.array((U_new[:, i]), ndmin=2))
                    mu_i_star_1 = U_new[:, i] * R[i, j] + mu_i_star_1

            Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
            Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V + epsilon * np.eye(Lambda_j_star_V.shape[0]))

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
            mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))


        V_new = np.transpose(np.reshape(V_new, (M, D)))


        U_old = np.array(U_new)
        V_old = np.array(V_new)


        if t > threshold:
            R_step = np.dot(np.transpose(U_new), V_new)
            for i in range(N):
                for j in range(M):
                    if R_step[i, j] > highest_rating:
                        R_step[i, j] = highest_rating
                    elif R_step[i, j] < lowest_rating:
                        R_step[i, j] = lowest_rating

            R_predict = (R_predict * (t - threshold - 1) + R_step) / (t - threshold)
            train_err = 0
            test_err = 0

            #train
            for i in range(N):
                for j in range(M):
                    if ranked(i, j):
                        train_err = train_err + (R_predict[i, j] - R[i, j]) ** 2
            train_err_list.append(np.sqrt(train_err / pairs_train))
            print("Training RMSE at iteration ", t - threshold, " :   ", "{:.4}".format(train_err_list[-1]))

            #test
            for i in range(N):
                for j in range(M):
                    if ranked_test(i, j):
                        test_err = test_err + (R_predict[i, j] - R_test[i, j]) ** 2
            test_err_list.append(np.sqrt(test_err / pairs_test))
            print("Test RMSE at iteration ", t - threshold, " :   ", "{:.4}".format(test_err_list[-1]))

            train_epoch_list.append(t)

    return R_predict, train_err_list, test_err_list, train_epoch_list






if __name__ == "__main__":
    N = 610
    M = 9742

    lowest_rating = 1
    highest_rating = 5

    print('Loading data...')

    datapath = 'dataset/ratings.csv'
    R_dataset, movieID = ranking_matrix(N, M, datapath)
    mask = np.random.choice([1, 0], size=(N, M), p=[0.9, 0.1])
    R = R_dataset * mask
    R_test = R_dataset * (1 - mask)

    # (N, M) = R.shape
    # print("There are {} users and {} movies".format(N, M))

    T = 50
    threshold = 0
    D_list = [50, 60, 70, 80]

    R_pred_list = []
    train_err_list = []
    test_err_list = []

    for D in D_list:
        print('hidden feature =', D)

        U_in = np.zeros((D, N))
        V_in = np.zeros((D, M))

        R_pred, train_err, test_err, train_epochs = BPMF(R, R_test, U_in, V_in, T, D, threshold, lowest_rating,
                                                         highest_rating)

        epochs = range(1, T)
        # plt.plot(epochs,train_err,label=f"{D} features train error")
        plt.plot(epochs, test_err, label=f"{D} features test error")
        plt.legend()

        R_pred_list.append(R_pred)
        train_err_list.append(train_err)
        test_err_list.append(test_err)
    plt.savefig(f" features error rate")
    plt.show()
