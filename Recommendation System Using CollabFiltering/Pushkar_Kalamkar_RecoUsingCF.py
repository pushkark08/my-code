from pyspark import SparkContext
import math
import operator
import time
import sys

start = time.time()
# user_dict = {}
sc = SparkContext(appName="RecommendationCF")

# read training file and remove headers
training_file = sys.argv[1]
trainer = sc.textFile(training_file)
f = trainer.first()
trainer = trainer.filter(lambda q: q != f)

# read testing file and remove headers
testing_file = sys.argv[2]
tester = sc.textFile(testing_file)
w = tester.first()
tester = tester.filter(lambda q: q != w)

train_data_movie = trainer.map(lambda l: (int(l.split(',')[1]), (int(l.split(',')[0]), float(l.split(',')[2])))).groupByKey()\
    .mapValues(dict).collectAsMap()
# train_data_user = trainer.map(lambda l: (int(l.split(',')[0]), (int(l.split(',')[1]), float(l.split(',')[2])))).groupByKey()\
#     .mapValues(dict).collectAsMap()
test_data_user = tester.map(lambda l: (int(l.split(',')[0]), (int(l.split(',')[1]), float(l.split(',')[2]))))\
    .groupByKey().sortByKey().mapValues(dict).collectAsMap()
test_data_movie = tester.map(lambda l: (int(l.split(',')[1]), (int(l.split(',')[0]), float(l.split(',')[2]))))\
    .groupByKey().mapValues(dict).collectAsMap()

# compute the averages for movies
averages = {}
for keys in train_data_movie:
    count = 0
    summ = 0
    for k in train_data_movie[keys]:
        count += 1
        summ += train_data_movie[keys][k]
    averages[keys] = summ/count

# compute the similarity matrix
similarity_matrix = {}
for i in train_data_movie:
    similarity_matrix[i] = {}
    for j in train_data_movie:
        if i != j:
            num = 0.0
            denom1 = 0.0
            denom2 = 0.0
            corated = train_data_movie[i].viewkeys() & train_data_movie[j].viewkeys()
            for k in corated:
                ri_avg = train_data_movie[i][k] - averages[i]
                rj_avg = train_data_movie[j][k] - averages[j]
                num += ri_avg*rj_avg
                denom1 += ri_avg**2
                denom2 += rj_avg**2
            denom = math.sqrt(denom1)*math.sqrt(denom2)
            if denom != 0:
                similarity_matrix[i][j] = num/denom
            else:
                similarity_matrix[i][j] = 0.0

# find the predictions
neighbours = 2000
predictions = {}
sqr = 0
count = 0
count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
# c = 0
for movie in test_data_movie:
    predictions[movie] = {}
    for user in test_data_movie[movie]:
        # c += 1
        predicted = 3.0
        if movie in similarity_matrix:
            sorted_sim_scores = sorted(similarity_matrix[movie].items(), key=operator.itemgetter(1), reverse=True)
            some_sim_scores = sorted_sim_scores[:neighbours]
            num = 0.0
            denom = 0.0
            for s in some_sim_scores:
                if user in train_data_movie[s[0]]:
                    num += s[1] * train_data_movie[s[0]][user]
                    denom += abs(s[1])
            if denom > 0:
                predicted = num/denom
        predictions[movie][user] = predicted
        actual = test_data_movie[movie][user]
        diff = abs(predicted - actual)
        sqr += (predicted - actual) ** 2
        count += 1
        if 0 <= diff < 1:
            count0 += 1
        elif 1 <= diff < 2:
            count1 += 1
        elif 2 <= diff < 3:
            count2 += 1
        elif 3 <= diff < 4:
            count3 += 1
        elif diff >= 4:
            count4 += 1

# print c
# print len(predictions)

# writing to file
out = open("output.txt", 'w')
out.write("UserId,MovieId,Pred_rating\n")
for x in test_data_user:
    for y in sorted(test_data_user[x]):
        out.write(str(x) + "," + str(y) + "," + str(predictions[y][x]) + '\n')
out.close()

# printing to console
print ">=0 and <1: ", count0
print ">=1 and <2: ", count1
print ">=2 and <3: ", count2
print ">=3 and <4: ", count3
print ">=4: ", count4
print "RMSE = ", (sqr/count) ** 0.5

print time.time() - start
