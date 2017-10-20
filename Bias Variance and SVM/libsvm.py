import scipy.io
from svmutil import svm_train
from svmutil import svm_predict

test_data = scipy.io.loadmat("phishing-test.mat")
features = test_data['features']
labels = list(test_data['label'][0])
x = [list(f) for f in features]

g_val = 0.25
c_val = 4096              # selected from prev.
m = svm_train(labels, x, '-t 2 -g %s -c %d' % (g_val, c_val))
labels, acc, val = svm_predict(labels, x, m)

print "The classification accuracy is: ", acc[0]
