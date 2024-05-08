import streamlit as st
import pandas as pd

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
#...
st.text("最佳分配方法：")
result=pd.DataFrame(index=["分销商"+str(i+1) for i in range(n)],columns=["商品"+str(i+1) for i in range(m)])
st.dataframe(result)
st.text("最大分配量：，最小费用：。")
st.divider()

st.header('批量上传数据（适用于大规模数据）：')
st.write("（文件格式与前述相同）")
price=st.number_input('请输入单位运费：',min_value=1,max_value=100,key=1)
st.write('单位运费：{}'.format(price))
data=st.file_uploader("Upload your file here!", type="csv")
#...
df=pd.read_csv(".gitignore")
csv=convert_df(df)
st.download_button(label="点击下载最佳分配方案",data=csv,file_name="solution.csv",mime='text/csv')
st.text("最大分配量：，最小费用：。")
st.divider()

button=st.button('完结撒花！')
if button:
    st.balloons()