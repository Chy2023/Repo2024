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
    dis:np.ndarray=field(init=False)
    vis:np.ndarray=field(init=False)
    cost:List[float]=field(init=False)
    gamma:float=0.99
    
    def __post_init__(self):
        C=copy.deepcopy(self.C)
        self.C=np.zeros((self.m+self.n+2,self.m+self.n+2))
        self.C[0,1:self.m+1]=self.V
        self.C[1:self.m+1,self.m+1:self.m+self.n+1]=C.T
        self.C[self.m+1:self.m+self.n+1,self.m+self.n+1]=self.U
        self.F=np.zeros((self.m+self.n+2,self.m+self.n+2))
        self.cur=np.zeros(self.m+self.n+2,dtype='int')
        self.G[0]=list(range(1,self.m+1))
        for i in range(1,self.m+1):
            self.G[i]=[0]+list(range(self.m+1,self.m+self.n+1))
        for i in range(self.m+1,self.m+self.n+1):
            self.G[i]=list(range(1,self.m+1))+[self.m+self.n+1]
        self.G[self.m+self.n+1]=list(range(self.m+1,self.m+self.n+1))
        self.vis=np.zeros(self.m+self.n+2)
        self.dis=np.zeros(self.m+self.n+2)
        sums=np.sum(C,axis=1)
        total=sum(sums)
        cost=[self.price*pow(self.gamma,int(self.n*i/total)) for i in sums]
        cost=np.expand_dims(cost,1).repeat(self.m,1)
        self.cost=np.zeros((self.m+self.n+2,self.m+self.n+2))
        self.cost[1:self.m+1,self.m+1:self.m+self.n+1]=cost.T
        
    def SPFA(self):
        INF=int(1e6)
        self.vis[:]=0
        self.dis[:]=INF
        q=queue.Queue(self.m+self.n+2)
        q.put(0)
        self.vis[0]=1
        self.dis[0]=0
        while not q.empty():
            u=q.get()
            self.vis[u]=0
            for v in self.G[u]:
                c=self.cost[u,v]
                if self.dis[v]>self.dis[u]+c and self.C[u,v]>self.F[u,v]:
                    self.dis[v]=self.dis[u]+c
                    if self.vis[v]==0:
                        q.put(v)
                        self.vis[v]=1
        self.cur[:]=0
        return (self.dis[self.m+self.n+1]!=INF)
    
    def DFS(self,u,cp):
        if u==self.m+self.n+1:
            self.vis[self.m+self.n+1]=1
            return cp
        flow=0
        for i in range(self.cur[u],len(self.G[u])):
            self.cur[u]+=1
            v=self.G[u][i]
            c=self.cost[u,v]
            if self.dis[u]+c==self.dis[v] and self.C[u,v]>self.F[u,v] and (self.vis[v]==0 or v==self.m+self.n+1):
                f=self.DFS(v,min(cp,self.C[u,v]-self.F[u,v]))
                if f>0:
                    print(u,v,f)
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
        while self.SPFA():
            tf=self.DFS(0,INF)
            if tf>0:
                sum+=tf
        X=self.F[1:self.m+1,self.m+1:self.m+self.n+1].T
        cost=np.sum(np.multiply(X,self.cost[1:self.m+1,self.m+1:self.m+self.n+1].T))
        return X,sum,cost
    
sol=Solution(3,3,10,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
print(sol.Dinic())