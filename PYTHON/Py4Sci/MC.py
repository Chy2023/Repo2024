from dataclasses import dataclass,field
from typing import List
import numpy as np
import copy

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
        MAX_IT=int(1e5)
        X=np.zeros((self.n,self.m)).astype(int).tolist()
        for _ in range(MAX_IT):
            result,cost=0,0
            V=copy.deepcopy(self.V)
            U=copy.deepcopy(self.U)
            for i in range(self.n):
                weight=weights[i]
                for j in range(self.m):
                    bound=min(self.C[i][j],self.V[j],self.U[i])
                    if bound==0:
                        X[i][j]=0
                        continue
                    X[i][j]=np.random.randint(low=0,high=bound+1)
                    V[j]-=X[i][j]
                    U[i]-=X[i][j]
                    result+=X[i][j]
                    cost+=X[i][j]*weight
            if result>self.sum or (result==self.sum and cost<self.cost):
                self.sum=result
                self.cost=cost
                self.X=copy.deepcopy(X)           

sol=Solution(3,3,10,0.99,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
sol.Solve()
print(sol.X)
print(sol.sum)
print(sol.cost)
