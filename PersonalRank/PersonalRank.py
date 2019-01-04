import time

def PersonalRank(G,alpha,root,max_depth):
    rank=dict()
    rank={x:0 for x in G.keys()}
    rank[root]=1
    # start iteration
    begin=time.time()
    for k in range(max_depth):
        tmp={x:0 for x in G.keys()}
        # get node i and its out nodes set ri
        for i,ri in G.items():
            print('----')
            # get node i's out node j and its edge weight wij, edge weight set defalut 1，nomalized to 1/len(ri)
            for j,wij in ri.items():
                tmp[j]+=alpha*rank[i]/(1.0*len(ri))
                # print('|'.join([str(i),str(j),str(tmp[j])]))
        tmp[root]+=(1-alpha)
        rank=tmp
    end=time.time()
    print ('use_time',end-begin)
    lst=sorted(rank.items(),key=lambda x:x[1],reverse=True)
    for ele in lst:
        print ("%s:%.3f, \t" %(ele[0],ele[1]))
 
    return rank
 
if __name__=='__main__':
    alpha=0.8
    G = {'A': {'a': 1, 'c': 1},
         'B': {'a': 1, 'b': 1, 'c': 1, 'd': 1},
         'C': {'c': 1, 'd': 1},
         'a': {'A': 1, 'B': 1},
         'b': {'B': 1},
         'c': {'A': 1, 'B': 1, 'C': 1},
         'd': {'B': 1, 'C': 1}}
    PersonalRank(G,alpha,'b',50)