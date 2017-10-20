import sys
import time
import itertools
from _collections import defaultdict
from pyspark import SparkContext


def algo(iterator):
    all_baskets = []
    unique_items_set = set()
    for b in iterator:
        for i in b[1]:
            unique_items_set.add(i)
        all_baskets.append(b[1])
    returned_frequent_items = set()
    sample = 1.0
    sample_baskets = all_baskets[:int(len(all_baskets) * sample)]
    sample_support = int(support_partition * sample)
    cand_items = set()
    sample_unique_set = set()
    count_dict = {}

    for basket in sample_baskets:
        for item in basket:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1

    for it in count_dict:
        if count_dict[it] >= sample_support:
            sample_unique_set.add(it)
            returned_frequent_items.add((it, 1))

    combinations = set(itertools.combinations(sample_unique_set, 2))
    for c in combinations:
        count = 0
        for basket in sample_baskets:
            if set(c).issubset(basket):
                count += 1
        if count >= sample_support:
            cand_items.add(frozenset(c))
            returned_frequent_items.add((frozenset(c), 1))
    sample_unique_set = cand_items
    k = 3
    while len(sample_unique_set) != 0:
        cand_items = set()
        temp_set = set()
        cand_dict = defaultdict(lambda: 0)
        kc2 = k*(k-1)/2
        combinations = set(itertools.combinations(sample_unique_set, 2))
        for c in combinations:
            grp = set().union(*c)
            if len(grp) == k:
                s = ''
                for g in sorted(grp):
                    s += str(g)
                cand_dict[s] += 1
                if cand_dict[s] == kc2:
                    temp_set.add(frozenset(grp))
        for it in temp_set:
            count = 0
            for basket in sample_baskets:
                if it.issubset(basket):
                    count += 1
            if count >= sample_support:
                cand_items.add(it)
                returned_frequent_items.add((it, 1))
        sample_unique_set = cand_items
        k += 1

    # print "map phase 1 output: ", len(returned_frequent_items)
    return iter(returned_frequent_items)

start = time.time()
sc = SparkContext(appName="Frequent_Itemsets")

file_name = sys.argv[2]
lines = sc.textFile(file_name)
f = lines.first()
lines = lines.filter(lambda q: q != f)

support = int(sys.argv[3])
case_number = int(sys.argv[1])


if case_number == 1:
    data = lines.map(lambda l: (int((l.split(',')[0])), int(l.split(',')[1]))).groupByKey().mapValues(set)
elif case_number == 2:
    data = lines.map(lambda l: (int((l.split(',')[1])), int(l.split(',')[0]))).groupByKey().mapValues(set)
baskets = data.collect()

num_of_partitions = data.getNumPartitions()
support_partition = support / num_of_partitions

data1 = data.mapPartitions(algo).reduceByKey(lambda a, b: a + b).collect()


def map_phase2_algo(iterator):
    map_phase2_result = []
    cand_list = []
    for i in iterator:
        cand_list.append(i[1])
    for i in data1:
        count = 0
        if isinstance(i[0], int):
            for basket in cand_list:
                if i[0] in basket:
                    count += 1
        else:
            for basket in cand_list:
                if i[0].issubset(basket):
                    count += 1
        map_phase2_result.append((i[0], count))
    return iter(map_phase2_result)

data2 = data.mapPartitions(map_phase2_algo).reduceByKey(lambda a, b: a + b).collect()

final_freq_dict = defaultdict(lambda: [])
for d in data2:
    if d[1] >= support:
        if isinstance(d[0], int):
            final_freq_dict[1].append(d[0])
        else:
            final_freq_dict[len(d[0])].append(sorted(list(d[0])))

total_count = 0

outfile = open('output.txt', 'w')
for k in sorted(final_freq_dict.keys()):
    first = True
    for v in sorted(final_freq_dict[k]):
        total_count += 1
        if isinstance(v, int):
            if first:
                outfile.write('(' + str(v) + ')')
                first = False
            else:
                outfile.write(',(' + str(v) + ')')
        else:
            internal_first = True
            in_list = ''
            for a in v:
                if internal_first:
                    in_list += str(a)
                    internal_first = False
                else:
                    in_list += ',' + str(a)
            if first:
                outfile.write('(' + in_list + ')')
                first = False
            else:
                outfile.write(',(' + in_list + ')')
    outfile.write('\n')
outfile.close()
print "Case: ", case_number, " Support: ", support
print "Total freq items: ", total_count
print time.time() - start
