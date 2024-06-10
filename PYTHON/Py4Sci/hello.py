import streamlit as st
import pandas as pd
from LP_v3 import *
from PIL import Image

@ st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.title('商品调配问题求解器')
st.subheader("——Chy的2024Py4Sci大作业")
add_selectbox = st.sidebar.selectbox(
    "请问你的身份是？",
    ("Unknown","USTCer", "游客")
)
if add_selectbox=="Unknown":
    st.sidebar.text("Robot?")
elif add_selectbox=="USTCer":
    st.sidebar.text("废理兴工!")
else:
    st.sidebar.text("Welcome!")
st.markdown(body=r'''# 问题背景：
&emsp;&emsp;考虑如下实际问题：假如你是某品牌电脑的总代理商，每次推出新品，你都需要将库存的电脑分配给若干下游代理商。电脑分配问题的具体条件如下：

1. **电脑可按照颜色、内存、CPU等分为若干种机型，每种机型分别有若干库存量。**
2. **每个代理商都对每种机型提出了各自的需求量，但每个代理商获得的总量不能超过给定的分配上限。**
3. **运输电脑需要总代理商支付额外的运输费用，运往不同代理商所需的单位运输费用可能不同。**

&emsp;&emsp;给定上述条件，你的优化目标如下：

1. **如何确定分配策略，使得总代理商分配给下游代理商的电脑总量最多？**
2. **如何在实现优化目标(1)的同时，使得总代理商需要支付的运输费用最少？**

&emsp;&emsp;此问题实际上属于**运筹学**和**算法设计**的交叉领域，而我这学期恰好修读了这两门专业课，后文将分别使用这两种学科的方法解决此问题。

# 实验目的：
&emsp;&emsp;分别使用上文提到的**运筹学(代数角度)**、**算法设计(几何角度)** 方法求解商品调配问题的最优方案，并分析两种方法的优缺点，完成学科交叉与讨论。同时，为了提高程序的可移植性，也为了彰显本课程强调的$Python$编程能力，**求解程序已制作成网页并上传至云服务器**，以供在线求解问题(网页链接：[*求解器*](https://repo2024-dfiqzv8hgeodcmtvegc6zf.streamlit.app/))。

# 实验环境：
$$
\begin{aligned}
&开发环境：\\
&VSCode\quad Python-3.11.7(conda)\\
&工具库：\\
&dataclasses,typing(格式处理)\\
&numpy,pandas,copy(数据处理)\\
&scipy(科学求解)\\
&queue(数据结构)\\
&PIL(图片处理)\\
&streamlit(制作网页)\\
&运行环境：\\
&StreamlitCloud(云服务器)\\
\end{aligned}
$$

# 问题建模：
&emsp;&emsp;为了形式化地求解此类问题，将上述情形抽象为以下的模型：

&emsp;&emsp;假设共有$m$种机型、$n$个代理商，第$j$种机型的库存量为$V_j$，第$i$个代理商的分配上限为$U_i$，第$i$个代理商对第$j$种机型的需求量为$C_{ij}$，将第$j$种机型运往第$i$个代理商的单位运输费用为$w_{ij}$。同时假设总代理商实际上向第$i$个代理商分配第$j$种机型的数目为$X_{ij}$ $(1\leq i\leq n$，$1\leq j\leq m)$。

&emsp;&emsp;为了简化问题，假设不同机型运往同一代理商的单位运输费用相同，即$w_{ij}=w_i$。为了优先满足配额高的代理商的需求，假设单位运输费用的原价均为$price$，折扣率为$\gamma=0.99$，则$w_i=price*\gamma^{\lfloor\frac{\sum_{j=1}^m C_{ij}}{\frac{1}{n}\sum_{i=1}^n \sum_{j=1}^m C_{ij}} \rfloor}$。直观上来看，如果某一代理商的需求量相对于其他代理商越多，那么其单位运输费用的折扣力度就越大。

&emsp;&emsp;优化目标如下：
$$
\left\{
\begin{aligned}
&max~~Z=\sum_{i=1}^n \sum_{j=1}^m X_{ij}\\
&st.\\
&0\leq X_{ij}\leq C_{ij}\\
&\sum_{i=1}^n X{ij}\leq V_j\\
&\sum_{j=1}^m X{ij}\leq U_i\\
&X\in \mathbb{N}
\end{aligned}
\right.
~~(1)
$$
$$
\left\{
\begin{aligned}
&min~~Y=\sum_{i=1}^n \sum_{j=1}^m w_{i}X_{ij}\\
&st.~~X=argmax~~Z
\end{aligned}
\right.
~~(2)
$$

# 求解方法：
## 线性规划：
&emsp;&emsp;从问题的建模中可以看出，该问题的约束条件和优化目标都是线性函数，于是可以很自然地使用线性规划法求解。

&emsp;&emsp;首先将原式写为矩阵形式，然后即可使用`scipy.optimize.linprog`求解问题：
$$
\begin{aligned}
&max~~Z=E^TX\\
&A_{ub}X\leq b_{ub}\\
&0\leq X\leq C\\
\end{aligned}~~(3)
$$
$$
\begin{aligned}
&X_{nm\times 1}=X_{n\times m}.flatten(),E_{nm\times 1}=(1,1,...,1)^T,(b_{ub})_{(n+m)\times 1}=(U^T,V^T)^T\\
&e_{1\times m}=(1,1,...,1),(A_{ub})_{(n+m)\times nm}=
\begin{bmatrix}
{e}&{O}&{\cdots}&{O}\\
{O}&{e}&{\cdots}&{O}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{O}&{O}&{\cdots}&{e}\\
{I_m}&{I_m}&{\cdots}&{I_m}\\
\end{bmatrix}
\end{aligned}
$$
$$
\begin{aligned}
&min~~Y=W^TX\\
&X=argmax~~Z~~~~~~~(4)\\
&W_{nm\times 1}=W_{n\times m}.flatten()\\
\end{aligned}
$$
&emsp;&emsp;实际操作时发现，`linprog`函数的功能有局限性：只能求解目标函数的最小值、不能同时求解多个目标函数。对于第一个问题，将$max~Z$换成$min~-Z$即可求解最大值；对于第二个问题，可采取下面两种方法：
1. 将两个目标函数加权相加，并求解新的目标函数，即$min~\alpha Z_1+(1-\alpha)Z_2,0<\alpha<1$。从直观上来看，$\alpha$的大小反映了两个目标函数的相对重要程度，$\alpha$越大，对$Z_1$的优化相对越强，对$Z_2$的优化相对越弱。对于本问题，应将$\alpha$取大些。
2. 先求解优化目标(3)，再将$Z$的值作为新的约束条件求解优化目标(4)。

&emsp;&emsp;上述即是线性规划法的求解过程，该方法的优缺点将在稍后讨论。
## 图论算法：
&emsp;&emsp;上一小节完全从代数方程的角度考虑问题，我们不妨转换思路，从几何关系的角度重新考虑该问题。

&emsp;&emsp;将原问题建模成图网络如下：

''')
image=Image.open('PYTHON/Py4Sci/graph.jpeg')
st.image(image,caption='图网络')
st.markdown(r'''&emsp;&emsp;图中共分为源点、机型、代理商、汇点四层结构，节点的值表示编号，存货量通过源点到机型的边控制，代理商配额上限通过代理商到汇点的边控制，代理商需求、运输费用由机型到代理商的边控制，每条边的流量表示商品的配额。

&emsp;&emsp;直观的理解：可将每个点看作水龙头，每条边看作水管；除了源点和汇点外，每个水龙头的流入量等于流出量，且水管中的流量不能超过上限。

&emsp;&emsp;**优化目标**：
1. 满足图网络流量限制的前提下，最大化从源点流出的总流量。
2. 满足优化目标(1)的前提下，最小化流量在图网络中的总运输费用。

&emsp;&emsp;由上图的建模表示，可以看出优化目标(1)实际上就是[*最大流问题*](https://zhuanlan.zhihu.com/p/103132159)。最大流问题有多种解法，而本问题最适合采用***Dinic算法***。

&emsp;&emsp;**预备知识**：
1. 层次网络：根据从源点到某节点的最短路径长度，将节点分层。可以看出，本问题的图网络天然具有层次网络的结构。
2. 残留网络：设图中的任意一条弧为$<u,v>$，则该弧的容量=容量-流量，同时增加反向弧$<v,u>$，反向弧的容量=流量。直观上来说，反向弧实际上代表“反悔”机制：过去选择了某条弧，现在可以选择其反向弧来撤销过去的选择。
''')
image=Image.open(r'PYTHON/Py4Sci/image-3.png')
st.image(image,caption='残差图')
st.markdown(r'''&emsp;&emsp;Dinic算法的求解步骤如下：
1. 初始化网络及流量。
2. 通过BFS构造残留网络、层次网络，若汇点不在层次网络中，则结束。
3. 通过DFS在层次网络中进行增广。
4. 转步骤(2)。

&emsp;&emsp;同理，可以看出优化目标(2)实际上就是[*最小费用最大流问题*](https://zhuanlan.zhihu.com/p/677980352)。类似于优化目标(1)，我们可以采用***SPFA+Dinic算法***。

&emsp;&emsp;只需将Dinic算法步骤(2)的BFS改为SPFA即可。解释如下：SPFA算法与Dijkstra算法都是求解最短路径的方法，但SPFA算法可以求解带负权值边的图；每次通过SPFA求解各点到源点的最短路径，再用DFS进行增广。

# 性能分析：
&emsp;&emsp;设共有$n$个变量，$m$个约束方程，以下分析两种问题解法的时间复杂度。

&emsp;&emsp;线性规划问题采用**单纯形法**求解：线性规划问题的定义域为单纯形，而最优解一定在单纯形的某个顶点处取得，因此每次从一个顶点转移到相邻的更优的顶点。单纯形共有$C_n^m$个顶点，最坏情况下可能遍历每一个顶点，复杂度可达到指数级别。幸运的是，可以证明单纯形算法的平均复杂度是多项式时间的。(***Borgwardt,1982***)

&emsp;&emsp;最小费用最大流问题采用**SPFA+Dinic算法**求解，复杂度为$O(n^2m)$，即使在最坏情况下也仅为多项式时间复杂度。

&emsp;&emsp;实际运行时发现，线性规划法的运行速度更快。这可能是因为最坏情况很少出现，也可能是因为`scipy`库由更高效的编程语言编写而成。

# 总结与收获：
&emsp;&emsp;线性规划法调用`scipy`库求解问题，求解速度很快，但是可能出现非整数解(这不符合问题的实际意义)，而对其取整后的结果却不一定是最优整数解。图论法采用自己编写的`Dinic`算法，求解速度稍慢，但得到的解一定是最优整数解。两种算法都只能求解问题的某一个最优解，而不能求出所有的最优解。

&emsp;&emsp;考虑将原问题做进一步拓展：假设一共有$n$级代理商，总代理商需要将货物分配给一级代理商，一级代理商又要分配给二级代理商...甚至同级代理商之间还可以串货。此时并不适合采用线性规划法，因为约束条件太多、太杂乱；而Dinic算法依然适用，因为约束条件可以很好地使用图网络表示。

&emsp;&emsp;总之，线性规划法是线性问题的通用解法，而Dinic算法是针对最小代价最大流这类问题的特定解法，二者各有优劣。

&emsp;&emsp;通过商品调配问题的求解，不仅将**运筹学**和**算法设计**这两门课的内容有机地结合起来，而且使得我们能从代数、几何两个角度看待同一个问题。同时，我在编写程序、设计网页的过程中，也提高了个人的代码能力。我想，这就是**面向科学问题求解的编程实践**这门课想要传递的精神。''')
st.markdown("# 求解器：")
st.success("Wanna take a try?Input your data below!")
st.header('手动输入数据（仅用于小规模数据）：')
m=st.number_input('请输入商品种类的数目：',min_value=2,max_value=6)
st.write('商品种类数：{}'.format(m))
n=st.number_input('请输入分销商的数目：',min_value=2,max_value=6)
st.write('分销商总数：{}'.format(n))
price=st.number_input('请输入单位运费：',min_value=1,max_value=100)
st.write('单位运费：{}'.format(price))
st.text("请在下方表格中继续填写相关数据：")
index=["分销商"+str(i+1) for i in range(n)]+["商品存量"]
columns=["商品"+str(i+1) for i in range(m)]+["分销上限"]
df=pd.DataFrame(index=index,columns=columns)
df=st.data_editor(df)

button=st.button('开始求解')
if button:
    V=df.iloc[n,0:m].to_numpy(dtype='int')
    U=df.iloc[0:n,m].to_numpy(dtype='int')
    C=df.iloc[0:n,0:m].to_numpy(dtype='int')
    print("V:{}".format(V))
    print("U:{}".format(U))
    print("C:{}".format(C))
    sol=Solution(m,n,price,V,U,C)
    X,b_eq,cost=sol.Solve()
    st.success('求解成功！')
    st.text("最佳分配方法：")
    result=pd.DataFrame(index=["分销商"+str(i+1) for i in range(n)],columns=["商品"+str(i+1) for i in range(m)])
    result.iloc[0:n,0:m]=X
    st.dataframe(result)
    st.text("最大分配量：{}，最小费用：{}。".format(b_eq,cost))
    st.divider()

st.header('批量上传数据（适用于大规模数据）：')
st.write("（文件格式与前述相同）")
price=st.number_input('请输入单位运费：',min_value=1,max_value=100,key=1)
st.write('单位运费：{}'.format(price))
data=st.file_uploader("Upload your file here!", type="csv")

button=st.button('开始求解',key=10)
if button:
    df=pd.read_csv(data,header=0,index_col = 0)
    m=len(df.columns)-1
    n=len(df)-1
    V=df.iloc[n,0:m].to_numpy(dtype='int')
    U=df.iloc[0:n,m].to_numpy(dtype='int')
    C=df.iloc[0:n,0:m].to_numpy(dtype='int')
    sol=Solution(m,n,price,V,U,C)
    X,b_eq,cost=sol.Solve()
    st.success('求解成功！')
    result=pd.DataFrame(index=["分销商"+str(i+1) for i in range(n)],columns=["商品"+str(i+1) for i in range(m)])
    result.iloc[0:n,0:m]=X
    csv=convert_df(result)
    st.download_button(label="点击下载最佳分配方案",data=csv,file_name="solution.csv",mime='text/csv')
    st.text("最大分配量：{}，最小费用：{:.2f}。".format(b_eq,cost))
    
st.divider()

url='https://github.com/Chy2023/Repo2024'
st.markdown(f'''<a href={url}><button style="background-color:GreenYellow;">我的Github仓库</button></a>''',unsafe_allow_html=True)
button=st.button('完结撒花！')
if button:
    st.balloons()