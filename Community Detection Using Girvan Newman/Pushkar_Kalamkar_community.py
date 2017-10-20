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

node_degrees = defaultdict(lambda: 0)
graph_dict = defaultdict(list)
for u in itertools.combinations(user_movie.keys(), 2):
    if len(set(user_movie[u[0]]).intersection(set(user_movie[u[1]]))) >= 3:
        graph_dict[u[0]].append(u[1])
        graph_dict[u[1]].append(u[0])
        node_degrees[u[0]] += 1
        node_degrees[u[1]] += 1


def bfs(root):
    parents = {root: []}                    # list of parents of node
    details = {root: [0, 1]}                # [length of shortest path to node, no. of shortest paths]
    processing_list = [root]                # list of current nodes
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
                else:
                    if details[edge_node][0] - details[node][0] == 1:         # if it is true parent
                        temp = parents[edge_node]
                        temp.append(node)
                        parents[edge_node] = temp
                        details[edge_node][1] += details[node][1]
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
# print "Betweenness Done!"

sorted_values = sorted(main_between.items(), key=operator.itemgetter(1), reverse=True)

communities = {1: [x for x in graph_dict.keys()]}
max_modularity = -100.0
max_communities = {}
iterations = 1
flag = True
for s in range(len(sorted_values)):
    if iterations > 5:
        break
    else:
        (e1, e2), v = sorted_values[s]
        edge = {e1, e2}
        temp = graph_dict[e1]
        temp.remove(e2)
        graph_dict[e1] = temp
        temp2 = graph_dict[e2]
        temp2.remove(e1)
        graph_dict[e2] = temp2
        l = len(main_between)
        for k in communities.keys():
            if edge.issubset(set(communities[k])):
                # bfs on e1
                if flag:
                    flag = False
                    modularity = 0.0
                    for c in communities:
                        for i in communities[c]:
                            for j in communities[c]:
                                if i != j:
                                    aij = 0
                                    ki_kj = node_degrees[i] * node_degrees[j]
                                    if i in graph_dict[j]:
                                        # if frozenset([i, j]) in main_between:
                                        aij = 1
                                    modularity += (aij - (ki_kj / (2.0 * l)))
                    modularity /= float(2 * l)
                    print "No. of communities: ", len(communities), " Modularity: ", modularity
                    if modularity > max_modularity:
                        max_modularity = modularity
                        max_communities = dict(communities)
                c_list = {e1}
                visited = {e1}
                while len(c_list) != 0:
                    temp_list = set()
                    for c in c_list:
                        for g in graph_dict[c]:
                            if g not in visited:
                                temp_list.add(g)
                                visited.add(g)
                    c_list = temp_list
                if e2 not in visited:
                    old_c = set(communities[k]) - visited
                    communities[k] = visited
                    communities[len(communities.keys()) + 1] = old_c
                    # calculate modularity
                    modularity = 0.0
                    for c in communities:
                        for i in communities[c]:
                            for j in communities[c]:
                                if i != j:
                                    aij = 0
                                    ki_kj = node_degrees[i] * node_degrees[j]
                                    if i in graph_dict[j]:
                                    # if frozenset([i, j]) in main_between:
                                        aij = 1
                                    modularity += (aij - (ki_kj/(2.0*l)))
                    iterations += 1
                    modularity /= float(2*l)
                    prev_modularity = modularity
                    print "No. of communities: ", len(communities), " Modularity: ", modularity
                    if modularity > max_modularity:
                        max_modularity = modularity
                        max_communities = dict(communities)
                break

# print "Max Modularity: ", max_modularity
# print "Optimal communities: ", max_communities

outfile_b = sys.argv[2]
out_b = open(outfile_b, 'w')
community_list = []
for m in max_communities:
    community_list.append(sorted(max_communities[m]))
community_list = sorted(community_list)
for p in community_list:
    out_b.write('[')
    fir = True
    for p1 in p:
        if fir:
            out_b.write(str(p1))
            fir = False
        else:
            out_b.write(',' + str(p1))
    out_b.write(']\n')
out_b.close()
print time.time() - start
