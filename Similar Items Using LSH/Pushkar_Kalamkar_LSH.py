from pyspark import SparkContext
from pyspark.sql import SparkSession
import itertools
import numpy as np
import time
import sys

start = time.time()
sc = SparkContext(appName="LSH")
s = SparkSession(sc)
training_file = sys.argv[1]
trainer = sc.textFile(training_file)
f = trainer.first()
trainer = trainer.filter(lambda q: q != f)
movie_user = trainer.map(lambda l: (int(l.split(',')[1]), int(l.split(',')[0]))).groupByKey()\
    .mapValues(list).collectAsMap()
movies = sorted(movie_user.keys())


def hash_f(h, x):
    q = 3
    a = h + 2
    return (a*x + q) % 671

hash_dict = {}
for k in range(0, 105):
    hash_dict[k] = {}
    for i in movie_user:
        mini = float('inf')
        for j in movie_user[i]:
            h_value = hash_f(k, j)
            if h_value < mini:
                mini = h_value
        hash_dict[k][i] = mini

col_dict = {}
for r in range(9066):
    col_dict[r] = movies[r]

matr = np.array([[hash_dict[i][j] for j in sorted(hash_dict[i])] for i in sorted(hash_dict)])

bands = 35
rows = 3
band_num = 0
combined_list = set()

while band_num < bands:
    indices = [range(band_num * rows, (band_num + 1) * rows)]
    b = matr[indices, :][0]
    dicti = {}
    for j in range(len(b[0])):
        t = tuple(b[:, j])
        if t in dicti:
            l = dicti[t]
            l.append(j)
            dicti[t] = sorted(l)
        else:
            dicti[t] = [j]
    for k in dicti.values():
        if len(k) > 1:
            combined_list.add(tuple(k))
    band_num += 1

pre_cand_list = set()
tp_list = set()

for e in combined_list:
    pairs = itertools.combinations(e, 2)
    for p in pairs:
        t = []
        for pa in p:
            t.append(col_dict[pa])
        c1 = set(movie_user[t[0]])
        c2 = set(movie_user[t[1]])
        js = float(len(c1.intersection(c2)))/float(len(c1.union(c2)))
        t.append(js)
        if js >= 0.5:
            tup = tuple(t)
            pre_cand_list.add(tup)
pre_cand_list = sorted(pre_cand_list)
# print len(pre_cand_list)

outfile = sys.argv[2]
out = open(outfile, "w")
for p in range(len(pre_cand_list)):
    out.write(str(pre_cand_list[p][0]) + "," + str(pre_cand_list[p][1]) + "," + str(pre_cand_list[p][2]) + "\n")
out.close()

# ground_truth = sc.textFile("SimilarMovies.GroundTruth.05.csv")
# ground_sets = ground_truth.map(lambda l: (int(l.split(',')[0]), int(l.split(',')[1]))).collect()
# print len(ground_sets)
# print "Recall: ", float(len(pre_cand_list))/float(len(ground_sets))

print time.time() - start
