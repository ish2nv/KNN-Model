import random
import pandas as pd
import numpy as np
from collections import Counter


def k_nearest_neighbors(adultdata, predictions_test,k):
  for k in range(k,39,2):
      How_Far = []
      for section in adultdata:
          for features in adultdata[section]:
              eucliden_distans = np.linalg.norm(np.array(features) - np.array(predictions_test))
              How_Far.append([eucliden_distans, section])
      list = [i[1] for i in sorted(How_Far)[:k]]
      vote_result = Counter(list).most_common(1)[0][0]
      return vote_result
  return k


df= pd.read_csv("data.new.csv")
dataList = df.astype(float).values.tolist()
random.shuffle(dataList)
split_testdata = 0.3

train_set = {0:[], 1:[]}
test_set = {0:[] , 1:[]}
train_dataset = dataList[:-int(split_testdata*len(dataList))]
test_dataset = dataList[-int(split_testdata*len(dataList)):]

for i in train_dataset:
    train_set[i[-1]].append(i[:-1])

for i in test_dataset:
    test_set[i[-1]].append(i[:-1])

matches = 0
comparisions =0

for sections in test_set:
    for data in test_set[sections]:
        vote = k_nearest_neighbors(train_set, data,k=1)
        if sections == vote:
            matches +=1
        comparisions += 1
print('Accuracy Rate:', matches/comparisions)

