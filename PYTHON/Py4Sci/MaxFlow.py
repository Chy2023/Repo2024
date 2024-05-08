from dataclasses import dataclass,field
from typing import Dict,List
import numpy as np
import copy
import queue

@dataclass
class Solution:
    m:int
    n:int
    price:int
    V:np.ndarray
    U:np.ndarray
    C:np.ndarray
    F:np.ndarray=field(init=False)
    G:Dict=field(default_factory=dict)
    cur:np.ndarray=field(init=False)
    level:np.ndarray=field(init=False)
    weights:List[float]=field(init=False)
    gamma:float=0.99
    
    def __post_init__(self):
        C=copy.deepcopy(self.C)
        self.C=np.zeros((self.m+self.n+2,self.m+self.n+2))
        self.C[0,1:self.m+1]=self.V
        self.C[1:self.m+1,self.m+1:self.m+self.n+1]=C.T
        self.C[self.m+1:self.m+self.n+1,self.m+self.n+1]=self.U
        self.level=np.zeros(self.m+self.n+2)
        self.F=np.zeros((self.m+self.n+2,self.m+self.n+2))
        self.cur=np.zeros(self.m+self.n+2,dtype='int')
        self.G[0]=list(range(1,self.m+1))
        for i in range(1,self.m+1):
            self.G[i]=[0]+list(range(self.m+1,self.m+self.n+1))
        for i in range(self.m+1,self.m+self.n+1):
            self.G[i]=list(range(1,self.m+1))+[self.m+self.n+1]
        self.G[self.m+self.n+1]=list(range(self.m+1,self.m+self.n+1))
        sums=np.sum(C,axis=1)
        total=sum(sums)
        self.weights=[self.price*pow(self.gamma,int(self.n*i/total)) for i in sums]

    def BFS(self):
        q=queue.Queue(self.m+self.n+2)
        self.level[:]=0
        q.put(0)
        self.level[0]=1
        while not q.empty():
            u=q.get()
            for v in self.G[u]:
                if (not self.level[v]) and self.C[u,v]>self.F[u,v]:
                    self.level[v]=self.level[u]+1
                    q.put(v)
        self.cur[:]=0
        return (self.level[self.m+self.n+1]!=0)
    
    def DFS(self,u,cp):
        if u==self.m+self.n+1 or cp==0:
            return cp
        flow=0
        for i in range(self.cur[u],len(self.G[u])):
            self.cur[u]+=1
            v=self.G[u][i]
            if self.level[u]+1==self.level[v] and self.C[u,v]>self.F[u,v]:
                f=self.DFS(v,min(cp,self.C[u,v]-self.F[u,v]))
                if f>0:
                    self.F[u,v]+=f
                    self.F[v,u]-=f
                    cp-=f
                    flow+=f
                    if cp==0:
                        break
        return flow
    
    def Dinic(self):
        sum=tf=0
        INF=int(1e6)
        while self.BFS():
            tf=self.DFS(0,INF)
            if tf>0:
                sum+=tf
        X=self.F[1:self.m+1,self.m+1:self.m+self.n+1].T
        cost=0
        for i in range(self.n):
            weight=self.weights[i]
            for j in range(self.m):
                cost+=X[i,j]*weight
        return X,sum,cost

sol=Solution(3,3,10,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
print(sol.Dinic())