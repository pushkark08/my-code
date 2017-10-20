import hw_utils
import time

training_x, training_y, test_x, test_y = hw_utils.loaddata('MiniBooNE_PID.txt')
# training_x, training_y, test_x, test_y = hw_utils.loaddata('sample_input.txt')
norm_train_x, norm_test_x = hw_utils.normalize(training_x, test_x)

d_in = 50
d_out = 2
archi_d1 = [[d_in, d_out], [d_in, 50, d_out], [d_in, 50, 50, d_out], [d_in, 50, 50, 50, d_out]]
archi_d2 = [[d_in, 50, d_out], [d_in, 500, d_out], [d_in, 500, 300, d_out], [d_in, 800, 500, 300, d_out], [d_in, 800, 800, 500, 300, d_out]]
start_time1 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_d2, actfn='linear', last_act='softmax',
                    reg_coeffs=[0.0],num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
end_time1 = time.time()
time_taken1 = end_time1 - start_time1
print "Time for Linear: ", time_taken1

start_time2 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_d2, actfn='sigmoid', last_act='softmax',
                    reg_coeffs=[0.0],num_epoch=30, batch_size=1000, sgd_lr=0.001, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
end_time2 = time.time()
time_taken2 = end_time2 - start_time2
print "Time for Sigmoid: ", time_taken2

start_time3 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_d2, actfn='relu', last_act='softmax',
                    reg_coeffs=[0.0],num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
end_time3 = time.time()
time_taken3 = end_time3 - start_time3
print "Time for ReLu: ", time_taken3

archi_g = [[d_in, 800, 500, 300, d_out]]
start_time4 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_g, actfn='relu', last_act='softmax',
                    reg_coeffs=[10**-7, 5*(10**-7), 10**-6, 5*(10**-6), (10**-5)],num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=False, verbose=0)
end_time4 = time.time()
time_taken4 = end_time4 - start_time4
print "Time for L2 with ReLu: ", time_taken4

start_time5 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_g, actfn='relu', last_act='softmax',
                    reg_coeffs=[10**-7, 5*(10**-7), 10**-6, 5*(10**-6), (10**-5)],num_epoch=30, batch_size=1000, sgd_lr=0.0005, sgd_decays=[0.0], sgd_moms=[0.0],
                    sgd_Nesterov=False, EStop=True, verbose=0)
end_time5 = time.time()
time_taken5 = end_time5 - start_time5
print "Time for L2 with ReLu and Early Stopping: ", time_taken5

start_time6 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_g, actfn='relu', last_act='softmax',
                    reg_coeffs=[5*(10**-7)],num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=[10**-5, 5*(10**-5), 10**-4, 3*(10**-4), 7*(10**-4), 10**-3],
                    sgd_moms=[0.0], sgd_Nesterov=False, EStop=False, verbose=0)
end_time6 = time.time()
time_taken6 = end_time6 - start_time6
print "Time for SGD with Wt decay: ", time_taken6

start_time7 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_g, actfn='relu', last_act='softmax',
                    reg_coeffs=[0.0],num_epoch=50, batch_size=1000, sgd_lr=0.00001, sgd_decays=[0.0001],
                    sgd_moms=[0.99, 0.98, 0.95, 0.9, 0.85], sgd_Nesterov=True, EStop=False, verbose=0)
end_time7 = time.time()
time_taken7 = end_time7 - start_time7
print "Time for Momentum ", time_taken7

start_time8 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_g, actfn='relu', last_act='softmax',
                    reg_coeffs=[10**-7], num_epoch=100, batch_size=1000, sgd_lr=0.00001, sgd_decays=[0.0001],
                    sgd_moms=[0.99], sgd_Nesterov=True, EStop=True, verbose=0)
end_time8 = time.time()
time_taken8 = end_time8 - start_time8
print "Time for Combining ", time_taken8


start_time9 = time.time()
hw_utils.testmodels(norm_train_x, training_y, norm_test_x, test_y, archs=archi_d2, actfn='relu', last_act='softmax',
                    reg_coeffs=[10**-7, 5*(10**-7), 10**-6, 5*(10**-6), 10**-5], num_epoch=100, batch_size=1000,
                    sgd_lr=0.00001, sgd_decays=[0.00001, 0.00005, 0.0001],
                    sgd_moms=[0.99], sgd_Nesterov=True, EStop=True, verbose=0)
end_time9 = time.time()
time_taken9 = end_time9 - start_time9
print "Time for Last part ", time_taken9
