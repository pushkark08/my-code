import scipy.stats as scstats
import itertools as itr
import numpy as np
import matplotlib.pyplot as mplot
import random
from sklearn.datasets import load_boston
boston_data = load_boston()

attribute_lists = boston_data['data']
test_data = []
train_data = []
test_target = []
train_target = []


def create_test_train_data():           # Train data is a 433 X 13 array
    for i in range(0, len(attribute_lists)):
        if i % 7 == 0:
            test_data.append(list(attribute_lists[i]))
            test_target.append(boston_data['target'][i])
        else:
            train_data.append(list(attribute_lists[i]))
            train_target.append(boston_data['target'][i])

create_test_train_data()
test_attribute_array = []
all_train_attribute_array = []


def create_arrays_for_hist():
    for j in range(0, len(train_data[0])):
        train_attribute_array = []
        for i in train_data:
            train_attribute_array.append(i[j])
        all_train_attribute_array.append(train_attribute_array)
create_arrays_for_hist()
pearson = []


def show_histograms():
    attributes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    his = mplot.figure()
    his.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    for i in range(13):
        his.add_subplot(5, 3, i + 1).hist(all_train_attribute_array[i])
        his.add_subplot(5, 3, i + 1).set_title(attributes[i])
        r_value, p_value = scstats.pearsonr(all_train_attribute_array[i], train_target)
        pearson.append(r_value)
    mplot.show()

show_histograms()
print "Pearson Co-efficients: ", pearson


def normalize_train_test():
    mean_train = np.mean(train_data, axis=0)
    std_train = np.std(train_data, axis=0, ddof=1)
    for i in range(0, len(train_data)):
        for j in range(0, len(train_data[0])):
            train_data[i][j] = (train_data[i][j] - mean_train[j])/std_train[j]
    for i in range(0, len(test_data)):
        for j in range(0, len(test_data[0])):
            test_data[i][j] = (test_data[i][j] - mean_train[j]) / std_train[j]


normalize_train_test()
weights = []
weights_ridge = []


def find_w_linear():
    global weights
    x_bar = []
    for i in train_data:
        x_bar.append([1] + i)
    x_bar_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_bar_trans, x_bar)
    xtrans_y = np.dot(x_bar_trans, train_target)
    weights = np.linalg.solve(xtrans_dot_x, xtrans_y)
find_w_linear()


def mean_squared_error_train(wt):
    w0 = wt[0]
    y_subtract_sq_sum = 0
    for i in range(0, len(train_data)):
        y_predict = w0
        for j in range(0, 13):
            y_predict += wt[j+1] * train_data[i][j]
        y_subtract_sq_sum += (y_predict - train_target[i]) ** 2
    mse = y_subtract_sq_sum/len(train_data)
    print "MSE on Training Data: ", mse

print "\nLinear Regression:"
mean_squared_error_train(weights)


def mean_squared_error_test(wt):
    w0 = wt[0]
    y_subtract_sq_sum_test = 0
    for i in range(0, len(test_data)):
        y_predict_test = w0
        for j in range(0, 13):
            y_predict_test += wt[j+1] * test_data[i][j]
        y_subtract_sq_sum_test += (y_predict_test - test_target[i]) ** 2
    mse_test = y_subtract_sq_sum_test/len(test_data)
    print "MSE on Test Data: ", mse_test

mean_squared_error_test(weights)


def find_w_ridge(lamb):
    global weights_ridge
    x_bar = []
    for i in train_data:
        x_bar.append([1] + i[:13])
    #print len(x_bar[0]), len(x_bar)
    x_bar_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_bar_trans, x_bar)
    xtrans_dot_x_plus_lambda = np.add(xtrans_dot_x, lamb * np.identity(14))
    xtrans_y = np.dot(x_bar_trans, train_target)
    weights_ridge = np.linalg.solve(xtrans_dot_x_plus_lambda, xtrans_y)

print "\nRidge regression:"
print "Lambda = 0.01: "
find_w_ridge(0.01)
mean_squared_error_train(weights_ridge)
mean_squared_error_test(weights_ridge)
print "Lambda = 0.1: "
find_w_ridge(0.1)
mean_squared_error_train(weights_ridge)
mean_squared_error_test(weights_ridge)
print "Lambda = 1: "
find_w_ridge(1)
mean_squared_error_train(weights_ridge)
mean_squared_error_test(weights_ridge)

new_train_data = []
new_train_target = []
combined_training_data = train_data[:]


def combine_attributes_with_target():
    for i in range(0, len(train_data)):
        combined_training_data[i].append(train_target[i])

combine_attributes_with_target()


def separate_data():
    random.shuffle(combined_training_data)
    for i in range(0, len(combined_training_data), 43):
        n_train_data = []
        n_train_target = []
        if i > 300:
            break
        else:
            for k in range(0, 43):
                n_train_data.append(combined_training_data[i + k][:13])
                n_train_target.append(combined_training_data[i + k][13])
            new_train_data.append(n_train_data)
            new_train_target.append(n_train_target)
    for i in range(301, len(combined_training_data), 44):
        n_train_data = []
        n_train_target = []
        for k in range(0, 44):
            n_train_data.append(combined_training_data[i + k][:13])
            n_train_target.append(combined_training_data[i + k][13])
        new_train_data.append(n_train_data)
        new_train_target.append(n_train_target)

separate_data()
weights_ridge_cv = []


def find_w_ridge_cv(lamb, leave):
    global weights_ridge_cv
    x_bar = []
    y = []
    for i in range(0, len(new_train_data)):
        if i != leave:
            for j in range(0, len(new_train_data[i])):
                x_bar.append([1] + new_train_data[i][j])
    for k in range(0, len(new_train_target)):
        if k != leave:
            for l in range(0, len(new_train_target[k])):
                y.append(new_train_target[k][l])
    x_bar_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_bar_trans, x_bar)
    xtrans_dot_x_plus_lambda = np.add(xtrans_dot_x, lamb * np.identity(14))
    xtrans_y = np.dot(x_bar_trans, y)
    weights_ridge_cv = np.linalg.solve(xtrans_dot_x_plus_lambda, xtrans_y)
    return weights_ridge_cv


def mean_squared_error_cv(wts, left_one):
    w0 = wts[0]
    y_subtract_sq_sum_test = 0
    for i in range(0, len(new_train_data[left_one])):
        y_predict_test = w0
        for j in range(0, 13):
            y_predict_test += wts[j + 1] * new_train_data[left_one][i][j]
        y_subtract_sq_sum_test += (y_predict_test - new_train_target[left_one][i]) ** 2
    mse_test = y_subtract_sq_sum_test / len(new_train_data[left_one])
    #print "MSE on CV'd Test Data after leaving: ", left_one, "is: ", mse_test
    return mse_test.round(decimals=4)


def find_for_lambda(lam):
    mse_array = []
    for i in range(0, 10):
        w_cv = find_w_ridge_cv(lam, i)
        mse_array.append(mean_squared_error_cv(w_cv, i))
    mse_avg = np.mean(mse_array, axis=0)
    return mse_avg.round(decimals=4)

best_lambda = 0


def select_best_lambda():
    global best_lambda
    lamb_array = [0.01]
    min_mse = 100000
    for lambd in lamb_array:
        for i in range(0, 1000):
            mse = find_for_lambda(lambd + 0.01*i)
            if mse < min_mse:
                min_mse = mse
                best_lambda = lambd + 0.01*i

print "Running....."
select_best_lambda()
find_w_ridge(best_lambda)
print "\nMSE for Test data after CV: "
#print "Best lambda after searching for 1000 values: ", best_lambda
mean_squared_error_test(weights_ridge)
weights_4_features = []                      # For b part of 3.3, first only attr number 12. Then attr number 5, then 10
best_features_array = [12, 5, 10, 3]
best_4_train_data = []
best_4_test_data = []
new_attribute_train_array = []


def create_4_feature_data():
    for i in train_data:
        best_4_train_data.append([i[j] for j in best_features_array])
    for i in test_data:
        best_4_test_data.append([i[j] for j in best_features_array])

#create_4_feature_data()


def find_w_linear_with_4_features():
    global weights_4_features
    x_bar = []
    for i in best_4_train_data:
        x_bar.append([1] + i)
    x_bar_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_bar_trans, x_bar)
    xtrans_y = np.dot(x_bar_trans, train_target)
    weights_4_features = np.linalg.solve(xtrans_dot_x, xtrans_y)

#find_w_linear_with_4_features()
residue_array = []


def mean_squared_error_train_with_4(wt):
    w0 = wt[0]
    y_subtract_sq_sum = 0
    for i in range(0, len(best_4_train_data)):
        y_predict = w0
        for j in range(0, 4):
            y_predict += wt[j+1] * best_4_train_data[i][j]
        residue_array.append(train_target[i] - y_predict)
        y_subtract_sq_sum += (y_predict - train_target[i]) ** 2
    #print residue_array
    mse = y_subtract_sq_sum/len(train_data)
    #print "MSE on Training Data with 4 best features: ", mse
    return mse

#mean_squared_error_train_with_4(weights_4_features)


def mean_squared_error_test_with_4(wt):
    w0 = wt[0]
    y_subtract_sq_sum_test = 0
    for i in range(0, len(best_4_test_data)):
        y_predict_test = w0
        for j in range(0, 4):
            y_predict_test += wt[j+1] * best_4_test_data[i][j]
        y_subtract_sq_sum_test += (y_predict_test - test_target[i]) ** 2
    mse_test = y_subtract_sq_sum_test/len(test_data)
    #print "MSE on Test Data with 4 best features ", mse_test

#mean_squared_error_test_with_4(weights_4_features)
new_pearson = []


def find_pearson():
    new_array = [0, 1, 2, 4, 6, 7, 8, 9, 11]
    new_train_attribute_array1 = []
    for i in range(0, len(all_train_attribute_array[0])):
        new_train_attribute_array1.append([all_train_attribute_array[j][i] for j in new_array])
    new_train_attribute_array = np.transpose(new_train_attribute_array1)
    # print len(new_train_attribute_array), new_train_attribute_array[0], len(residue_array)
    for i in range(0, len(new_train_attribute_array)):
        r_val, p_val = scstats.pearsonr(new_train_attribute_array[i], residue_array)
        new_pearson.append(r_val)

#find_pearson()
#print "New Pearson Coeff 1:", new_pearson


def find_all_combinations():                                # for brute force search
    global best_features_array
    global best_4_train_data
    min_mse_train = 100000
    best_i = []
    all_combs = itr.combinations(list(range(0, 13)), 4)
    for i in all_combs:
        best_4_train_data = []
        best_features_array = i
        #print best_features_array,
        create_4_feature_data()
        find_w_linear_with_4_features()
        mse_train = mean_squared_error_train_with_4(weights_4_features)
        if mse_train < min_mse_train:
            min_mse_train = mse_train
            best_i = i
    print "\nBrute Force Search: "
    print "Best 4 feature numbers: ", best_i
    print "MSE: ", min_mse_train

find_all_combinations()

############################## For 3.4 polynomial expansion #############################

poly_train_data = []
poly_test_data = []


def create_poly_train_data():
    for i in range(len(train_data)):
        a = []
        for j in range(len(train_data[i]) - 1):
            for k in range(j, len(train_data[i]) - 1):
                a.append(train_data[i][j] * train_data[i][k])
        poly_train_data.append(a)


def create_poly_test_data():
    for i in range(len(test_data)):
        a = []
        for j in range(len(test_data[i])):
            for k in range(j, len(test_data[i])):
                a.append(test_data[i][j] * test_data[i][k])
        poly_test_data.append(a)

create_poly_train_data()
create_poly_test_data()


def standardize():
    a_mean = np.mean(poly_train_data, axis=0)
    a_std = np.std(poly_train_data, axis=0, ddof=1)
    for i in range(0, len(poly_train_data)):
        for j in range(0, len(poly_train_data[0])):
            poly_train_data[i][j] = (poly_train_data[i][j] - a_mean[j])/a_std[j]
    for i in range(0, len(poly_test_data)):
        for j in range(0, len(poly_test_data[0])):
            poly_test_data[i][j] = (poly_test_data[i][j] - a_mean[j]) / a_std[j]

standardize()

newer_train_data = []
for q in range(len(train_data)):
    newer_train_data.append(train_data[q][:13])
poly_train_data = [newer_train_data[r] + poly_train_data[r] for r in range(len(poly_train_data))]
poly_test_data = [test_data[r] + poly_test_data[r] for r in range(len(poly_test_data))]
weights_poly = []


def find_w_linear_poly():
    global weights_poly
    x_bar = []
    for i in poly_train_data:
        x_bar.append([1] + i)
    x_bar_trans = np.transpose(x_bar)
    xtrans_dot_x = np.dot(x_bar_trans, x_bar)
    xtrans_y = np.dot(x_bar_trans, train_target)
    weights_poly = np.linalg.solve(xtrans_dot_x, xtrans_y)

find_w_linear_poly()


def mean_squared_error_train_poly(wt):
    w0 = wt[0]
    y_subtract_sq_sum = 0
    for i in range(0, len(poly_train_data)):
        y_predict = w0
        for j in range(0, 104):
            y_predict += wt[j+1] * poly_train_data[i][j]
        y_subtract_sq_sum += (y_predict - train_target[i]) ** 2
    mse = y_subtract_sq_sum/len(poly_train_data)
    print "MSE on Training Data for Features Expansion: ", mse

print "\nPolynomial Feature Expansion:"
mean_squared_error_train_poly(weights_poly)


def mean_squared_error_test_poly(wt):
    w0 = wt[0]
    y_subtract_sq_sum_test = 0
    for i in range(0, len(poly_test_data)):
        y_predict_test = w0
        for j in range(0, 104):
            y_predict_test += wt[j+1] * poly_test_data[i][j]
        y_subtract_sq_sum_test += (y_predict_test - test_target[i]) ** 2
    mse_test = y_subtract_sq_sum_test/len(poly_test_data)
    print "MSE on Test Data after Feature Expansion : ", mse_test

mean_squared_error_test_poly(weights_poly)
