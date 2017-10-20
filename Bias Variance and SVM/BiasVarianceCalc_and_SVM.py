import matplotlib.pyplot as mplot
import numpy as np
import scipy.io as sc
import time


def get_y(n):
    x = np.random.uniform(-1, 1, n)
    noise = np.random.normal(0.0, 0.1, n)
    y = 2*(x**2) + noise
    return list(x), list(y)

datasets1 = []
datasets2 = []


def gen_datasets(size, datasets):
    for q in range(0, 100):
        data_list = []
        x, y = get_y(size)
        x0 = [i ** 0 for i in x]
        x2 = [i ** 2 for i in x]
        x3 = [i ** 3 for i in x]
        x4 = [i ** 4 for i in x]
        for g in range(0, len(x)):
            data_list.append([x0[g], x[g], x2[g], x3[g], x4[g], y[g]])
        datasets.append(data_list)


gen_datasets(10, datasets1)
weights = []


def find_w_linear(inp, target):
    global weights
    x_bar = np.array(inp)
    target_array = np.transpose(target)
    target_array = list(target_array)
    x_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_trans, x_bar)
    xtrans_y = np.dot(x_trans, target_array)
    weights = np.linalg.solve(xtrans_dot_x, xtrans_y)
    return weights


def mean_squared_error_train(wt, inp, target):
    y_subtract_sq_sum = 0
    for i in range(0, len(inp)):
        y_predict = 0
        for j in range(0, len(inp[i])):
            y_predict += wt[j] * inp[i][j]
        y_subtract_sq_sum += (y_predict - target[i]) ** 2
    mse = y_subtract_sq_sum/len(inp)
    return round(mse, 5)


def find_bias(classifier_num, all_weights, test_list_x, test_list_y):                 # For 1 classifier: bias is the error that the best hypothesis(among all datasets) has with y.
    opt_weights = np.average(all_weights, axis=0)           # optimal weights
    count = 0
    bias = 0
    for x in test_list_x:
        inp_array = []
        for i in range(0, classifier_num):
            x_array = x ** i
            inp_array.append(x_array)
        opt_y = 0
        for i in range(0, len(inp_array)):
            opt_y += opt_weights[i] * inp_array[i]
        bias += (opt_y - test_list_y[count]) ** 2
        count += 1
    return round(bias/len(test_list_x), 6)


def find_variance(classifier_num, all_weights, test_list_x):          # For 1 classifier: Variance is the diff of our hypothesis from optimal one
    variance = 0
    opt_weights = np.average(all_weights, axis=0)           # optimal weights
    for x in test_list_x:
        inp_array = []
        for i in range(0, classifier_num):
            x_array = x ** i
            inp_array.append(x_array)
        opt_y = 0
        for i in range(0, len(inp_array)):
            opt_y += opt_weights[i] * inp_array[i]
        s = 0
        for w in all_weights:
            y = 0
            for i in range(0, len(inp_array)):
                y += w[i] * inp_array[i]
            s += (opt_y - y) ** 2
        variance += s/len(all_weights)
    return round(variance/len(test_list_x), 7)


def find_mse(datasets):
    hist_array = []
    print "Classifier", '\t\t', "Bias", '\t\t\t\t', "Variance"

    b0 = 0
    for x in range(0, len(test_list_x)):
        b0 += (1 - test_list_y[x]) ** 2
    b0 /= len(test_list_x)
    print "g", 1, '\t\t\t', round(b0, 6), '\t\t\t', 0

    all_sse = []
    for i in datasets:
        y_subtract_sq_sum = 0
        for j in i:
            y_subtract_sq_sum += (1 - j[5]) ** 2
        sse = y_subtract_sq_sum/len(i)
        all_sse.append(sse)
    hist_array.append(all_sse)

    for i in range(1, 6):
        mse = []
        wts_array = []
        for j in datasets:
            input_array = []
            target_array = []
            for k in j:
                input_array.append(k[:i])
                target_array.append(k[5])
            wt = find_w_linear(input_array, target_array)
            wts_array.append(wt)
            mse.append(mean_squared_error_train(weights, input_array, target_array))
        b = find_bias(i, wts_array, test_list_x, test_list_y)
        v = find_variance(i, wts_array, test_list_x)
        print "g", i+1, '\t\t\t', b, '\t\t\t', v
        hist_array.append(mse)
    show_histograms(hist_array, len(datasets[0]))


def show_histograms(hist_ar, size):
    attributes = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6']
    his = mplot.figure()
    his.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    his.suptitle("Datasets of size %d"%size, fontsize=20)
    for i in range(6):
        his.add_subplot(3, 2, i + 1).hist(hist_ar[i])
        his.add_subplot(3, 2, i + 1).set_title(attributes[i])
    mplot.show()


test_list_x, test_list_y = get_y(500)
print "\nWhen each dataset contains 10 samples:"
find_mse(datasets1)
print "\nWhen each dataset contains 100 samples:"
gen_datasets(100, datasets2)
find_mse(datasets2)
weights_ridge = []


def find_w_ridge(l, inp, target):
    global weights_ridge
    target_array = np.transpose(target)
    target_array = list(target_array)
    x_bar = np.array(inp)
    x_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_trans, x_bar)
    xtrans_dot_x_plus_lambda = np.add(xtrans_dot_x, l * np.identity(3))
    xtrans_y = np.dot(x_trans, target_array)
    weights_ridge = np.linalg.solve(xtrans_dot_x_plus_lambda, xtrans_y)
    return weights_ridge


def find_ridge_bias_variance():
    lambda_array = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    for l in lambda_array:
        wts_array = []
        for j in datasets2:
            input_array = []
            target_array = []
            for k in j:
                input_array.append(k[:3])
                target_array.append(k[5])
            wt = find_w_ridge(l, input_array, target_array)
            wts_array.append(wt)
        b = find_bias(3, wts_array, test_list_x, test_list_y)
        v = find_variance(3, wts_array, test_list_x)
        print l, '\t\t\t', b, '\t\t\t', v

print '\nRidge Regression:'
print "Lambda", '\t\t\t', "Bias", '\t\t\t\t', "Variance"
find_ridge_bias_variance()


svm_train_data = sc.loadmat('phishing-train.mat')
svm_test_data = sc.loadmat('phishing-test.mat')
train_features = svm_train_data['features']
train_label = svm_train_data['label'][0]
test_features = svm_test_data['features']
test_label = svm_test_data['label'][0]

train_features_trans = np.transpose(train_features)
columns = [1, 6, 7, 13, 14, 25, 28]
cols = [0,2,3,4,5,8,9,10,11,12,16,17,19,20,21,22,23,24,26,27,29]


def preprocess_train():
    global train_features_trans
    for c in cols:
        for i in range(0, len(train_features_trans[c])):
            if train_features_trans[c][i] == -1:
                train_features_trans[c][i] = 0

    for c in columns:
        minus_one = []
        zero = []
        for i in range(0, len(train_features_trans[c])):
            if train_features_trans[c][i] == 1:
                zero.append(0)
                minus_one.append(0)
            elif train_features_trans[c][i] == 0:
                zero.append(1)
                minus_one.append(0)
            else:
                train_features_trans[c][i] = 0
                zero.append(0)
                minus_one.append(1)
        train_features_trans = np.append(train_features_trans, [zero], axis=0)
        train_features_trans = np.append(train_features_trans, [minus_one], axis=0)

preprocess_train()
train_features = np.transpose(train_features_trans)
train_features_list = list(list(z) for z in train_features)

cv_accuracies1 = []
cv_accuracies2 = []
cv_accuracies3 = []
from svmutil import svm_train

c_list = [4**-6, 4**-5, 4**-4, 4**-3, 4**-2, 4**-1, 1, 4, 4**2]
for c in c_list:
    start = time.time()
    cv_acc = svm_train(list(train_label), train_features_list, '-t 0 -c %f -v 3'%c)
    end = time.time()
    cv_accuracies1.append([cv_acc, round((end - start)/3, 4)])


max2 = 0
c_max2 = 0
d_max2 = 0
c_list2 = [4**-3, 4**-2, 4**-1, 1, 4, 4**2, 4**3, 4**4, 4**5, 4**6, 4**7]
d_list = [1, 2, 3]
for c in c_list2:
    for d in d_list:
        start = time.time()
        cv_acc = svm_train(list(train_label), train_features_list, '-d %d -t 1 -c %f -v 3' %(d, c))
        if cv_acc > max2:
            max2 = cv_acc
            c_max2 = c
            d_max2 = d
        end = time.time()
        cv_accuracies2.append([round(c, 5), d, cv_acc, round((end - start)/3, 4)])

max3 = 0
c_max3 = 0
g_max3 = 0
for c in c_list2:
    for g in list([4**-7, 4**-6, 4**-5, 4**-4, 4**-3, 4**-2, 4**-1]):
        start = time.time()
        cv_acc = svm_train(list(train_label), train_features_list, '-g %f -t 2 -c %f -v 3'%(g, c))
        if cv_acc > max3:
            max3 = cv_acc
            c_max3 = c
            g_max3 = g
        end = time.time()
        cv_accuracies3.append([round(c, 5), round(g, 5), cv_acc, round((end - start)/3, 4)])

print "\nC\t\t\tAccuracy\tAvg. Time"
for i in range(0, len(cv_accuracies1)):
    print round(c_list[i], 4), '\t\t', cv_accuracies1[i][0], '\t\t', cv_accuracies1[i][1]


print "\nC\t\t\t\tDegree\t\tAccuracy\tAvg. Time"
for i in range(0, len(cv_accuracies2)):
    print cv_accuracies2[i][0], '\t\t\t', cv_accuracies2[i][1], '\t\t', cv_accuracies2[i][2], '\t\t', cv_accuracies2[i][3]


print "\nC\t\t\t\tGamma\t\tAccuracy\tAvg. Time"
for i in range(0, len(cv_accuracies3)):
    print cv_accuracies3[i][0], '\t\t\t', cv_accuracies3[i][1], '\t\t', cv_accuracies3[i][2], '\t\t', cv_accuracies3[i][3]


print "\nPoly Best: ", max2, c_max2, d_max2
print "\nRBF Best: ", max3, c_max3, round(g_max3, 5)

