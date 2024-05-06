from dataclasses import dataclass,field
from typing import List
import numpy as np
from scipy import optimize

@dataclass
class Solution:
    m:int
    n:int
    price:int
    gamma:float
    V:np.ndarray
    U:np.ndarray
    C:np.ndarray
    X:np.ndarray=field(init=False)
    sum:int=0
    cost:float=0
    
    def __post_init__(self):
        self.X=np.zeros((self.n,self.m))
    
    def Solve(self):
        sums=np.sum(self.C,axis=1)
        total=sum(sums)
        weights=[self.price*pow(self.gamma,int(self.n*i/total)) for i in sums]
        c=np.ones(self.n*self.m)*(-1)
        bounds=np.array([[0,i] for i in self.C.flatten()])
        A=np.zeros((self.n,self.n*self.m))
        for i in range(self.n):
            A[i,i*self.m:i*self.m+self.m]=1
        B=np.concatenate([np.eye(self.m)]*self.n,axis=1)
        A_ub=np.append(A,B,axis=0)
        b_ub=np.append(self.U,self.V)
        res=optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')
        self.X=res.x.reshape(self.n,self.m)
        self.sum=res.fun*(-1)
        for i in range(self.n):
            weight=weights[i]
            for j in range(self.m):
                self.cost+=self.X[i,j]*weight
        print(self.X)
        print(self.sum)
        print(self.cost)

sol=Solution(3,3,10,0.99,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
sol.Solve()