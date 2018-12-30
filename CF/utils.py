import random
import operator
import math
import time

def read_data(fileDir = '../data/ml-100k/u.data'):
    data = []
    fp = open(fileDir,'r')
    for line in fp:
        user, item , _, __ = line.strip().split()
        data.append([user, item, 1.0]) # set default interest  1.0
    return data


def split_data(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user, item, rate in data:
        if random.randint(0, M-1) == k:
            test.append([user, item, rate])
        else:
            train.append([user, item, rate])
    return train, test


def recall(test,result, N):
    hit = 0
    all = 0
    for user in test.keys():
        tu = test[user]
        rank = result[user]
        for item, p in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)


def precision(test,result, N):
    hit = 0
    all = 0
    for user in test.keys():
        tu = test[user]
        rank = result[user]
        for item, p in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    return hit / (all * 1.0)


def coverage(train, result, N):
    all_items = set()
    rec_items = set()

    for user,items in train.items():
        for i, p in items.items():
            all_items.add(i)  

    for user,items in result.items():
        for i, p in items:
            rec_items.add(i)
    
    return len(rec_items) / (len(all_items) * 1.0 )


def popularity(train, result):
    N = dict()
    ret = 0
    n = 0

    for user,items in train.items():
        for i in items:
            N.setdefault(i,0)
            N[i] += 1
    
    for user,items in result.items():
        for i, p in items:
            n += 1
            if i in train:
                ret += math.log(N[i] + 1)
                
    
    return ret / n
        
        
def data_transform(data):
    """transform to dict formate"""
    ret = dict()
    for u, i, r in data:
        ret.setdefault(u,dict())
        ret[u][i] = r
        
    return ret

        