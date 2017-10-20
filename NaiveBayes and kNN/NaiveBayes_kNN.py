from scipy.spatial import distance
import numpy
import _heapq as heap
import collections
import math

training_dict = {}
test_dict = {}
l1_dist = {}
class_list = [0, 0, 0, 0, 0, 0, 0]
glass_train_set = open('train.txt', 'r')
while True:
    line = glass_train_set.readline()
    if line is not '':
        line = line.strip()
        columns = line.split(',')
        lis = []
        for i in range(1,len(columns)-1):
            lis.append(float(columns[i]))
        lis.append(int(columns[len(columns)-1]))
        training_dict[columns[0]] = lis
        class_list[lis[len(lis) - 1] - 1] += 1
    else:
        break


glass_test_set = open('test.txt', 'r')
while True:
    line = glass_test_set.readline()
    if line is not '':
        line = line.strip()
        columns = line.split(',')
        lis = []
        for i in range(1,len(columns)-1):
            lis.append(float(columns[i]))
        lis.append(int(columns[len(columns)-1]))
        test_dict[columns[0]] = lis
    else:
        break


class_dict = {}
main_array = []
for j in training_dict.keys():
    end_length = len(training_dict[j])
    array_to_be_passed = training_dict[j][:end_length-1]
    main_array.append(array_to_be_passed)
    class_num = training_dict[j][end_length - 1]
    if class_num not in class_dict.keys():
        biggest_array = []
        biggest_array.append(array_to_be_passed)
        class_dict[class_num] = biggest_array
    elif class_num in class_dict.keys():
        class_dict[class_num].append(array_to_be_passed)


arr = numpy.array(main_array)
arr_mean = numpy.mean(arr, axis=0)                          # all means of attributes
arr_std = numpy.std(arr, axis=0, ddof=1)                    # all std of attributes
number_of_train_data = len(training_dict.keys())

mean_dict = {}
std_dict = {}


def find_class_means():
    for it in range(1, 8):
        if it in class_dict.keys():
            cl_array = numpy.array(class_dict[it])
            array_mean = numpy.mean(cl_array, axis=0)
            mean_dict[it] = array_mean
            array_std = numpy.std(cl_array, axis=0, ddof=1)
            std_dict[it] = array_std

find_class_means()
number_of_test_data = len(test_dict.keys())

def calc_percentages(c):
    return round(100*float(c)/number_of_train_data, 2)


def calc_percentages_test(c):
    return round(100*float(c)/number_of_test_data, 2)


def get_gaussian(x, mean, sd):
    var = float(sd)**2
    denominator = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denominator


def get_value(array, class_lab):
    val = 1.0
    mean_list = mean_dict[class_lab]
    std_list = std_dict[class_lab]
    for r in range(0, len(array)-1):
        if std_list[r] == 0:
            if mean_list[r] == array[r]:
                #val += 0
                val *= 1
            else:
                #val = 0
                val *= 0
                return val
        else:
            #val += math.log(get_gaussian(array[r], mean_list[r], std_list[r]))
            val *= get_gaussian(array[r], mean_list[r], std_list[r])
    return val


def get_prob_class(c):
    return float(class_list[c-1])/number_of_train_data
    #return math.log(float(class_list[c-1])/number_of_train_data)


def bayes_train():
    accuracy = 0
    error = 0
    for k in training_dict:
        found_label = None
        valu = None
        sample_arr = training_dict[k]
        original_label = sample_arr[len(sample_arr) - 1]
        for t in range(1, 8):
            if t == 4:
                continue
            prob = get_value(sample_arr, t)
            v = get_prob_class(t) * prob
            if valu is not None:
                if v > valu:
                    valu = v
                    found_label = t
            else:
                valu = v
                found_label = t
        if original_label != found_label:
            error += 1
        else:
            accuracy += 1
    print 'Training :', '\t\t', calc_percentages(accuracy), '\t\t', calc_percentages(error)


def bayes_test():
    accuracy = 0
    error = 0
    for k in test_dict:
        found_label = None
        valu = None
        sample_arr = test_dict[k]
        original_label = sample_arr[len(sample_arr) - 1]
        for t in range(1, 8):
            if t == 4:
                continue
            prob = get_value(sample_arr, t)
            v = get_prob_class(t) * prob
            if valu is not None:
                if v > valu:
                    valu = v
                    found_label = t
            else:
                valu = v
                found_label = t
        if original_label != found_label:
            error += 1
        else:
            accuracy += 1
    print 'Testing :', '\t\t', calc_percentages_test(accuracy), '\t\t', calc_percentages_test(error)

print 'Naive Bayes ', '\t', 'Accuracy', '\t', 'Error'
bayes_train()
bayes_test()
print


def normalize():
    for k in training_dict.keys():
        for m in range(0, len(training_dict[k])-1):
            training_dict[k][m] = (training_dict[k][m] - arr_mean[m])/arr_std[m]


def l1_metric(arr1, arr2):
    return sum(abs((a - b)) for a,b in zip(arr1, arr2))


def l2_metric(arr1, arr2):
    return distance.euclidean(arr1, arr2)


def max_label(heap):
    label_dict = {}
    for l in heap:
        label = l[1]
        val = l[0]
        if label_dict.get(label) is not None:
            label_dict[label][0] += 1
            label_dict[label][1] = min(label_dict[label][1], val)
        else:
            li = [1, val]
            label_dict[label] = li

    max_count = -1
    label = None
    min_d = None

    for key in label_dict.keys():
        if label_dict[key][0] > max_count:
            label = key
            min_d = label_dict[key][1]
            max_count = label_dict[key][0]

        elif label_dict[key][0] == max_count:
            if label_dict[key][1] < min_d:
                min_d = label_dict[key][1]
                label = key
    return label


def training(k):
    if k == 1:
        wrong_l1 = 0
        correct_l1 = 0
        wrong_l2 = 0
        correct_l2 = 0
        for key in training_dict:
            min_value_l1 = float('Inf')
            min_value_l2 = float('Inf')
            class_label_l1 = None
            class_label_l2 = None
            sample_array = training_dict[key]
            for tr_key in training_dict.keys():
                if tr_key == key:
                    continue
                compared_to_array = training_dict[tr_key]
                l1_distance = l1_metric(sample_array[:len(sample_array)-1], compared_to_array[:len(compared_to_array)-1])
                l2_distance = l2_metric(sample_array[:len(sample_array)-1], compared_to_array[:len(compared_to_array)-1])
                if l1_distance < min_value_l1:
                    min_value_l1 = l1_distance
                    class_label_l1 = compared_to_array[len(compared_to_array)-1]
                if l2_distance < min_value_l2:
                    min_value_l2 = l2_distance
                    class_label_l2 = compared_to_array[len(compared_to_array) - 1]
            original_label = sample_array[len(sample_array)-1]
            if original_label != class_label_l1:
                wrong_l1 += 1
            else:
                correct_l1 += 1
            if original_label != class_label_l2:
                wrong_l2 += 1
            else:
                correct_l2 += 1
        print 'K =', k
        print 'Metric', '\t', 'Accuracy', '\t', 'Error'
        print 'L1 :', '\t', calc_percentages(correct_l1), '\t\t', calc_percentages(wrong_l1)
        print 'L2 :', '\t', calc_percentages(correct_l2), '\t\t', calc_percentages(wrong_l2)
    else:
        wrong_l1_3 = 0
        correct_l1_3 = 0
        wrong_l2_3 = 0
        correct_l2_3 = 0
        for key in training_dict:
            sample_array = training_dict[key]
            lis_l1 = []
            lis_l2 = []
            desired_label_l1 = 0
            desired_label_l2 = 0
            for tr_key in training_dict.keys():
                if tr_key == key:
                    continue
                compared_to_array = training_dict[tr_key]
                l1_distance_3 = l1_metric(sample_array[:len(sample_array) - 1],
                                          compared_to_array[:len(compared_to_array) - 1])
                l2_distance_3 = l2_metric(sample_array[:len(sample_array) - 1],
                                          compared_to_array[:len(compared_to_array) - 1])
                class_label = compared_to_array[len(compared_to_array)-1]
                lis_l1.append([l1_distance_3, class_label])
                lis_l2.append([l2_distance_3, class_label])
            if k == 3:
                small_3_l1 = heap.nsmallest(3, lis_l1)
                small_3_l2 = heap.nsmallest(3, lis_l2)
            elif k == 5:
                small_3_l1 = heap.nsmallest(5, lis_l1)
                small_3_l2 = heap.nsmallest(5, lis_l2)
            elif k == 7:
                small_3_l1 = heap.nsmallest(7, lis_l1)
                small_3_l2 = heap.nsmallest(7, lis_l2)
            desired_label_l1 = max_label(small_3_l1)
            desired_label_l2 = max_label(small_3_l2)
            original_label = sample_array[len(sample_array) - 1]

            # label_list_l1 = []
            # dist_list_l1 = []
            # label_list_l2 = []
            # dist_list_l2 = []
            #
            # for h in range(0, len(small_3_l1)):
            #     label_list_l1.append(small_3_l1[h][1])
            #     dist_list_l1.append(small_3_l1[h][0])
            # c_l1 = collections.Counter(label_list_l1).most_common()
            # #print c_l1
            # flag_l1 = False
            # if k == 5:
            #     if list(c_l1[0])[1] == 2 and len(c_l1) == k - 2:
            #         #print small_3_l1
            #         #print list(c_l1[2])
            #         for w in small_3_l1:
            #             if w[1] == list(c_l1[2])[0]:
            #                 small_3_l1.remove(w)
            #                 flag_l1 = True
            #
            # if len(c_l1) > k - 1 or flag_l1:
            #     shortest_dist = min(dist_list_l1)
            #     for q in small_3_l1:
            #         if q[0] == shortest_dist:
            #             desired_label_l1 = q[1]
            # else:
            #     desired_label_l1 = c_l1[0][0]
            #
            # for h in range(0, len(small_3_l2)):
            #     label_list_l2.append(small_3_l2[h][1])
            #     dist_list_l2.append(small_3_l2[h][0])
            # c_l2 = collections.Counter(label_list_l2).most_common()
            # flag_l2 = False
            # if k == 5:
            #     if list(c_l2[0])[1] == 2 and len(c_l2) == k - 2:
            #         #print list(c_l2[2])
            #         for w1 in small_3_l2:
            #             if w1[1] == list(c_l2[2])[0]:
            #                 #print w1[1]
            #                 small_3_l2.remove(w1)
            #                 flag_l2 = True
            # if len(c_l2) > k - 1 or flag_l2:
            #     shortest_dist = min(dist_list_l2)
            #     for q in small_3_l2:
            #         if q[0] == shortest_dist:
            #             desired_label_l2 = q[1]
            # else:
            #     desired_label_l2 = c_l2[0][0]

            if original_label != desired_label_l1:
                wrong_l1_3 += 1
            else:
                correct_l1_3 += 1

            if original_label != desired_label_l2:
                wrong_l2_3 += 1
            else:
                correct_l2_3 += 1

        print 'K =', k
        print 'Metric', '\t', 'Accuracy', '\t', 'Error'
        print 'L1 :', '\t', calc_percentages(correct_l1_3), '\t\t', calc_percentages(wrong_l1_3)
        print 'L2 :', '\t', calc_percentages(correct_l2_3), '\t\t', calc_percentages(wrong_l2_3)


normalize()
print "KNN:"
print "Train Data:"
training(1)
training(3)
training(5)
training(7)


def normalize_test():
    for ki in test_dict.keys():
        for m in range(0, len(test_dict[ki])-1):
            test_dict[ki][m] = (test_dict[ki][m] - arr_mean[m])/arr_std[m]


#print number_of_test_data


def testing(k):
    if k == 1:
        wrong_l1 = 0
        correct_l1 = 0
        wrong_l2 = 0
        correct_l2 = 0
        for key in test_dict:
            min_value_l1 = float('Inf')
            min_value_l2 = float('Inf')
            class_label_l1 = None
            class_label_l2 = None
            sample_array = test_dict[key]
            for tr_key in training_dict.keys():
                compared_to_array = training_dict[tr_key]
                l1_distance = l1_metric(sample_array[:len(sample_array)-1], compared_to_array[:len(compared_to_array)-1])
                l2_distance = l2_metric(sample_array[:len(sample_array)-1], compared_to_array[:len(compared_to_array)-1])
                if l1_distance < min_value_l1:
                    min_value_l1 = l1_distance
                    class_label_l1 = compared_to_array[len(compared_to_array)-1]
                if l2_distance < min_value_l2:
                    min_value_l2 = l2_distance
                    class_label_l2 = compared_to_array[len(compared_to_array) - 1]
            original_label = sample_array[len(sample_array)-1]

            if original_label != class_label_l1:
                wrong_l1 += 1
            else:
                correct_l1 += 1
            if original_label != class_label_l2:
                wrong_l2 += 1
            else:
                correct_l2 += 1
        print 'K =', k
        print 'Metric', '\t', 'Accuracy', '\t', 'Error'
        print 'L1 :', '\t', calc_percentages_test(correct_l1), '\t\t', calc_percentages_test(wrong_l1)
        print 'L2 :', '\t', calc_percentages_test(correct_l2), '\t\t', calc_percentages_test(wrong_l2)
    else:
        wrong_l1_3 = 0
        correct_l1_3 = 0
        wrong_l2_3 = 0
        correct_l2_3 = 0
        for key in test_dict:
            sample_array = test_dict[key]
            lis_l1 = []
            lis_l2 = []
            desired_label_l1 = 0
            desired_label_l2 = 0
            for tr_key in training_dict.keys():
                compared_to_array = training_dict[tr_key]
                l1_distance_3 = l1_metric(sample_array[:len(sample_array) - 1],
                                          compared_to_array[:len(compared_to_array) - 1])
                l2_distance_3 = l2_metric(sample_array[:len(sample_array) - 1],
                                          compared_to_array[:len(compared_to_array) - 1])
                class_label = compared_to_array[len(compared_to_array)-1]
                lis_l1.append([l1_distance_3, class_label])
                lis_l2.append([l2_distance_3, class_label])
            if k == 3:
                small_3_l1 = heap.nsmallest(3, lis_l1)
                small_3_l2 = heap.nsmallest(3, lis_l2)
            elif k == 5:
                small_3_l1 = heap.nsmallest(5, lis_l1)
                small_3_l2 = heap.nsmallest(5, lis_l2)
            elif k == 7:
                small_3_l1 = heap.nsmallest(7, lis_l1)
                small_3_l2 = heap.nsmallest(7, lis_l2)
            desired_label_l1 = max_label(small_3_l1)
            desired_label_l2 = max_label(small_3_l2)
            original_label = sample_array[len(sample_array) - 1]

            # label_list_l1 = []
            # dist_list_l1 = []
            # label_list_l2 = []
            # dist_list_l2 = []
            #
            # for h in range(0, len(small_3_l1)):
            #     label_list_l1.append(small_3_l1[h][1])
            #     dist_list_l1.append(small_3_l1[h][0])
            # c_l1 = collections.Counter(label_list_l1).most_common()
            # #print c_l1
            # flag_l1 = False
            # if k == 5:
            #     if list(c_l1[0])[1] == 2 and len(c_l1) == k - 2:
            #         #print small_3_l1
            #         #print list(c_l1[2])
            #         for w in small_3_l1:
            #             if w[1] == list(c_l1[2])[0]:
            #                 small_3_l1.remove(w)
            #                 flag_l1 = True
            #
            # if len(c_l1) > k - 1 or flag_l1:
            #     shortest_dist = min(dist_list_l1)
            #     for q in small_3_l1:
            #         if q[0] == shortest_dist:
            #             desired_label_l1 = q[1]
            # else:
            #     desired_label_l1 = c_l1[0][0]
            #
            # for h in range(0, len(small_3_l2)):
            #     label_list_l2.append(small_3_l2[h][1])
            #     dist_list_l2.append(small_3_l2[h][0])
            # c_l2 = collections.Counter(label_list_l2).most_common()
            # flag_l2 = False
            # if k == 5:
            #     if list(c_l2[0])[1] == 2 and len(c_l2) == k - 2:
            #         #print list(c_l2[2])
            #         for w1 in small_3_l2:
            #             if w1[1] == list(c_l2[2])[0]:
            #                 #print w1[1]
            #                 small_3_l2.remove(w1)
            #                 flag_l2 = True
            # if len(c_l2) > k - 1 or flag_l2:
            #     shortest_dist = min(dist_list_l2)
            #     for q in small_3_l2:
            #         if q[0] == shortest_dist:
            #             desired_label_l2 = q[1]
            # else:
            #     desired_label_l2 = c_l2[0][0]

            if original_label != desired_label_l1:
                wrong_l1_3 += 1
            else:
                correct_l1_3 += 1

            if original_label != desired_label_l2:
                wrong_l2_3 += 1
            else:
                correct_l2_3 += 1
        print 'K =', k
        print 'Metric', '\t', 'Accuracy', '\t', 'Error'
        print 'L1 :', '\t', calc_percentages_test(correct_l1_3), '\t\t', calc_percentages_test(wrong_l1_3)
        print 'L2 :', '\t', calc_percentages_test(correct_l2_3), '\t\t', calc_percentages_test(wrong_l2_3)


normalize_test()
print
print "Test Data:"
print
testing(1)
testing(3)
testing(5)
testing(7)
