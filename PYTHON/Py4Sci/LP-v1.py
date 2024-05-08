from dataclasses import dataclass,field
from typing import List
import numpy as np
from scipy import optimize

@dataclass
class Solution:
    m:int
    n:int
    price:int
    V:np.ndarray
    U:np.ndarray
    C:np.ndarray
    gamma:float=0.99
    weights:List[float]=field(init=False)
    
    def __post_init__(self):
        sums=np.sum(self.C,axis=1)
        total=sum(sums)
        self.weights=[self.price*pow(self.gamma,int(self.n*i/total)) for i in sums]
    
    def Solve(self):
        c=np.ones(self.n*self.m)*(-1)
        bounds=np.array([[0,i] for i in self.C.flatten()])
        A=np.zeros((self.n,self.n*self.m))
        for i in range(self.n):
            A[i,i*self.m:i*self.m+self.m]=1
        B=np.concatenate([np.eye(self.m)]*self.n,axis=1)
        A_ub=np.append(A,B,axis=0)
        b_ub=np.append(self.U,self.V)
        res=optimize.linprog(c=c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,method='highs')
        X=res.x.reshape(self.n,self.m)
        result=res.fun*(-1)
        cost=0
        for i in range(self.n):
            weight=self.weights[i]
            for j in range(self.m):
                cost+=X[i,j]*weight
        print(X)
        print(result)
        print(cost)
        return X,result,cost

sol=Solution(3,3,10,[2,2,2],[2,2,2],np.array([[1,0,0],[0,1,1],[0,0,1]]))
sol.Solve()