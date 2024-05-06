from dataclasses import dataclass,field
from typing import List
import numpy as np

@dataclass
class Solution:
    m:int
    n:int
    price:int
    gamma:float
    V:List[int]
    U:List[int]
    C:List[List[int]]
    X:List[List[int]]=field(init=False)
    sum:int=0
    cost:float=0
    
    def __post_init__(self):
        self.X=np.zeros((self.n,self.m)).astype(int).tolist()

    def Solve(self):
        sums=np.sum(self.C,axis=1)
        total=sum(sums)
        weights=[self.price*pow(self.gamma,int(self.n*i/total)) for i in sums]
        d=dict(enumerate(sums))
        l=sorted(d.items(),key=lambda x:x[1],reverse=True)
        index,_=zip(*l)
        for i in index:
            weight=weights[i]
            req=self.C[i].copy()
            d=dict(enumerate(req))
            l=sorted(d.items(),key=lambda x:x[1],reverse=True)
            subindex,_=zip(*l)
            for j in subindex:
                self.X[i][j]=min(self.C[i][j],self.V[j],self.U[i])
                if self.X[i][j]==0:
                    continue
                self.V[j]-=self.X[i][j]
                self.U[i]-=self.X[i][j]
                self.sum+=self.X[i][j]
                self.cost+=self.X[i][j]*weight            

sol=Solution(3,3,10,0.99,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
sol.Solve()
print(sol.X)
print(sol.sum)
print(sol.cost)
