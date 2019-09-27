import math
import numpy as np
import math

def nDCG(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec

def auc(y_pred, y_test):
    pass
    

def pearson(self,rating1,rating2):
    """
    Pearson CorrelationCoefficient
        rating1：ratings of user1, {"movieid1":rate1,"movieid2":rate2,...}
        rating2：ratings of user2, {"movieid1":rate1,"movieid2":rate2,...}
    """
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    num = 0
    for key in rating1.keys():
        if key in rating2.keys():
            num += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += math.pow(x,2)
            sum_y2 += math.pow(y,2)
    if num == 0:
        return  0
    denominator = math.sqrt( sum_x2 - math.pow(sum_x,2) / num) * math.sqrt( sum_y2 - math.pow(sum_y,2) / num )
    if denominator == 0:
        return  0
    else:
        return ( sum_xy - ( sum_x * sum_y ) / num ) / denominator

if __name__ == '__main__':
    y_test = np.array([1,0,0,0,1,0,1,0,])
    y_pred = np.array([0.9, 0.8, 0.3, 0.1,0.4,0.9,0.66,0.7])
    print(y_test)
    print(y_pred)
