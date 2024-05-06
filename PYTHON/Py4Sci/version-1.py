from dataclasses import dataclass,field
from typing import List
import numpy as np

@dataclass
class Solution:
    m:int
    n:int
    V:List[int]
    U:List[int]
    C:List[List[int]]
    X:List[List[int]]=field(init=False)
    sum:int=0
    
    def __post_init__(self):
        self.X=np.zeros((self.n,self.m)).astype(int).tolist()
    
    
""" sol=Solution(3,3,[2,2,2],[2,2,2],[[1,1,1],[1,1,1],[1,1,1]])
print(sol.X) """