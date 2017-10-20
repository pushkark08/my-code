import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from scipy.stats import multivariate_normal as mn
import math

data = pd.read_csv('hw5_blob.csv', header=None)
c_data = pd.read_csv('hw5_circle.csv', header=None)


def make_centroids(size):
    centroid = []
    for i in range(size):
        ind = np.random.randint(0, len(data[0]))
        q = []
        for j in range(len(data.columns)):
            q.append(round(data[j][ind], 5))
        centroid.append(q)
    return centroid


def calc_distances(centroid, size):
    clusters = [[] for _ in range(size)]
    for i in range(len(data[0])):
        min_dist = float('inf')
        x_arr = np.array([data[0][i], data[1][i]])
        label = 0
        for j in range(size):
            cent_0 = np.array([centroid[j][0], centroid[j][1]])
            dist = np.linalg.norm(x_arr - cent_0)
            if dist < min_dist:
                label = j
                min_dist = dist
        clusters[label].append(x_arr)
    return clusters


def update_centroids(centroid, clust, size):
    max_iter = 100
    iter = 1
    cl = []
    while iter < max_iter:
        iter += 1
        new_centroids = []
        for i in range(size):
            m = np.mean(clust[i], axis=0)
            m = m.tolist()
            for j in range(len(m)):
                m[j] = round(m[j], 5)
            new_centroids.append(m)
        if centroid == new_centroids:
            break
        else:
            centroid = new_centroids
            cl = calc_distances(centroid, size)
    for i in range(size):
        c_axes = np.array(cl[i]).transpose()
        py.plot(c_axes[0], c_axes[1], '.')
    py.show()
    # print iter, centroid, len(cl), len(cl[0]), len(cl[1])


cent = make_centroids(2)
d = calc_distances(cent, 2)
update_centroids(cent, d, 2)

cent = make_centroids(3)
d = calc_distances(cent, 3)
update_centroids(cent, d, 3)

cent = make_centroids(5)
d = calc_distances(cent, 5)
update_centroids(cent, d, 5)


def dist_kernel(centroid, k_clusters):
    for i in range(len(k_data[0])):
        min_dist = float('inf')
        x_arr = np.array([k_data[0][i], k_data[1][i], k_data[2][i]])
        label = 0
        for j in range(2):
            # print centroid[0]
            cent_0 = np.array([centroid[j][0], centroid[j][1], centroid[j][2]])
            dist = np.linalg.norm(x_arr - cent_0)
            if dist < min_dist:
                label = j
                min_dist = dist
        k_clusters[label].append(x_arr)
    return k_clusters


def update_centres():
    clusters = [[] for _ in range(2)]
    centroid = []
    for i in range(2):
        ind = np.random.randint(0, len(k_data[0]))
        q = []
        for j in range(len(k_data.columns)):
            q.append(k_data[j][ind])
        centroid.append(q)
    # print centroid
    # centroid.append([k_data[0][ind], k_data[1][ind], k_data[2][ind]])
    # for w in range(2):
    k_clust = dist_kernel(centroid, clusters)
    max_iter = 100
    iter = 1
    while iter < max_iter:
        iter += 1
        new_centroids = []
        for i in range(2):
            m = np.mean(k_clust[i], axis=0)
            m = m.tolist()
            new_centroids.append(m)
        if centroid == new_centroids:
            original_clust = [[] for _ in range(2)]
            cl = clusters
            for a in range(2):
                for b in cl[a]:
                    original_points = [b[0]**0.5, b[1]**0.5]
                    for c in range(len(k_data)):
                        if original_points == [abs(c_data[0][c]), abs(c_data[1][c])]:
                            original_clust[a].append([c_data[0][c], c_data[1][c]])
            for a in range(2):
                orig = np.array(original_clust[a]).transpose()
                py.plot(orig[0], orig[1], '.')
            print iter
            py.show()
            break
        else:
            centroid = new_centroids


k_data = pd.DataFrame()
k_data[0] = [c_data[0][j] ** 2 for j in range(len(c_data[0]))]
k_data[1] = [c_data[1][k]**2 for k in range(len(c_data[0]))]
k_data[2] = [c_data[0][k]**2 + c_data[1][k]**2 for k in range(len(c_data[0]))]

update_centres()


def log_likelihood(size, alpha, mu, cov):
    overall_log_sum = 0
    for n in range(len(data)):
        summ = 0
        for q in range(size):
            summ += alpha[q] * mn.pdf([data[0][n], data[1][n]], mu[q], cov[q])
        log_sum = math.log(summ)
        overall_log_sum += log_sum
    return overall_log_sum


def e_step(size, alpha, mu, cov):
    wts = [[], [], []]
    for q in range(size):
        for i in range(len(data)):
            x = [data[0][i], data[1][i]]
            v = mn.pdf(x, mu[q], cov[q])
            wts[q].append(v * alpha[q])
    for i in range(len(wts[0])):
        summ = wts[0][i] + wts[1][i] + wts[2][i]
        for j in range(len(wts)):
            wts[j][i] = wts[j][i]/summ
    return wts


def m_step(size, c_wt, alpha, mu, cov):
    for q in range(size):
        wts_sum_k = 0
        xsum_for_mu = 0
        ysum_for_mu = 0
        for n in range(len(c_wt[q])):
            wts_sum_k += c_wt[q][n]
            xsum_for_mu += c_wt[q][n]*data[0][n]
            ysum_for_mu += c_wt[q][n] * data[1][n]
        alpha[q] = wts_sum_k/len(data)
        mu[q] = [xsum_for_mu/wts_sum_k, ysum_for_mu/wts_sum_k]
        sum_for_cov = [[0,0],[0,0]]
        for n in range(len(c_wt[q])):
            x_arr = np.array([data[0][n], data[1][n]])
            mu_arr = np.array(mu[q])
            sub = np.subtract(x_arr, mu_arr)
            sub_arr = np.matrix(sub)
            sub_arr_t = np.transpose(sub_arr)
            prod = np.dot(sub_arr_t, sub_arr)
            wts_prod = np.multiply(c_wt[q][n], prod)
            sum_for_cov = np.add(wts_prod, sum_for_cov)
        cov[q] = np.array(np.divide(sum_for_cov, wts_sum_k))


def run_em(size, alpha, mu, cov, log_lik):
    l = log_likelihood(size, alpha, mu, cov)
    itert = 0
    it = []
    all_l = []
    while abs(l - log_lik) > 0.0001:
        log_lik = l
        c_wts = e_step(size, alpha, mu, cov)
        m_step(size, c_wts, alpha, mu, cov)
        l = log_likelihood(size, alpha, mu, cov)
        all_l.append(l)
        it.append(itert+1)
        itert += 1
    py.plot(it, all_l)
    return l, c_wts
    # print mu, itert


def run_5_times():
    best_l = -10000
    best_mu = []
    best_cov = []
    best_weigh = []
    for r in range(5):
        cov_m = np.cov(data, rowvar=False)
        cov = [cov_m, cov_m, cov_m]
        alpha = [0.34, 0.33, 0.33]
        mu = []
        for b in range(3):
            ind = np.random.randint(0, len(k_data[0]))
            mu.append([data[0][ind], data[1][ind]])
        log_lik = 0
        l, weigh = run_em(3, alpha, mu, cov, log_lik)
        if l > best_l:
            best_l = l
            best_mu = mu
            best_cov = cov
            best_weigh = weigh
    print "Best Log likelihood: ", best_l
    print "Best mean: ", best_mu
    print "Best covariance matrix: ", best_cov
    py.show()
    cl_assign = [[], [], []]
    print len(best_weigh), len(best_weigh[0])
    for i in range(len(best_weigh[0])):
        best = np.array([best_weigh[0][i], best_weigh[1][i], best_weigh[2][i]])
        max_index = best.argmax()
        cl_assign[max_index].append([data[0][i], data[1][i]])
    print len(cl_assign), len(cl_assign[0]), len(cl_assign[1]), len(cl_assign[2])
    for a in range(3):
        orig = np.array(cl_assign[a]).transpose()
        py.plot(orig[0], orig[1], '.')
    py.show()


run_5_times()
