from pyspark import SparkContext
from _collections import defaultdict
import itertools
import operator
import time
import sys

start = time.time()
sc = SparkContext(appName="Community Detection")
training_file = sys.argv[1]
trainer = sc.textFile(training_file)
f = trainer.first()
trainer = trainer.filter(lambda q: q != f)
user_movie = trainer.map(lambda l: (int(l.split(',')[0]), int(l.split(',')[1]))).groupByKey()\
    .mapValues(list).collectAsMap()

graph_dict = defaultdict(list)
for u in itertools.combinations(user_movie.keys(), 2):
    if len(set(user_movie[u[0]]).intersection(set(user_movie[u[1]]))) >= 3:
        graph_dict[u[0]].append(u[1])
        graph_dict[u[1]].append(u[0])


def bfs(root):
    parents = {root: []}                    # list of parents of node
    details = {root: [0, 1]}                # [length of shortest path to node, no. of shortest paths]
    processing_list = [root]                # list of current nodes
    # leaf_list = [root]                    # list of leaf nodes
    levels = []                             # levels of tree
    betweenness = defaultdict(lambda: 1.0)
    while len(processing_list) != 0:
        child_nodes = set()
        for node in processing_list:
            for edge_node in graph_dict[node]:
                if edge_node not in parents:                                 # means not encountered
                    parents[edge_node] = [node]
                    details[edge_node] = [1+details[node][0], details[node][1]]
                    child_nodes.add(edge_node)
                    # leaf_list.append(edge_node)
                    # if node in leaf_list:
                    #     leaf_list.remove(node)
                else:
                    if details[edge_node][0] - details[node][0] == 1:         # if it is true parent
                        temp = parents[edge_node]
                        temp.append(node)
                        parents[edge_node] = temp
                        details[edge_node][1] += details[node][1]
                        # if node in leaf_list:
                        #     leaf_list.remove(node)
        levels.append(processing_list)
        processing_list = child_nodes
    for z in range(len(levels)-1, 0, -1):
        for l in levels[z]:
            par = parents[l]
            if len(par) > 0:
                for p in par:
                    betweenness[p] += betweenness[l] * (details[p][1] / float(details[l][1]))
                    main_between[frozenset([l, p])] += betweenness[l] * (details[p][1]/float(details[l][1]))

main_between = defaultdict(lambda: 0)
for i in graph_dict.keys():
    bfs(i)

outfile_b = sys.argv[2]
out_b = open(outfile_b, 'w')
out_list = []
for i in main_between:
    n1, n2 = i
    temp_tup = sorted((n1, n2))
    tup = (temp_tup, main_between[i]/2.0)
    out_list.append(tup)

sorted_x = sorted(out_list, key=operator.itemgetter(0), reverse=False)
for i in sorted_x:
    n, v = i
    n1, n2 = n
    out_b.write("(" + str(n1) + "," + str(n2) + "," + str(v) + ")" + '\n')
out_b.close()

print time.time() - start
