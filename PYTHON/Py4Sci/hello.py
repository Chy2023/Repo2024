import streamlit as st
import pandas as pd
from LP_v3 import *

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
st.header('问题背景：')
st.divider()
#。。。
st.header('求解方法：')
st.divider()
#。。。
st.header('优劣分析：')
st.divider()
#。。。
st.header('结果讨论：')
st.divider()

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