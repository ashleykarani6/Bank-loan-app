import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go



st.title('Bank Loan Payment Classifier')

df = pd.read_csv("df.csv")

if st.checkbox('Show the dataframe'):
    st.write(df.head(5))


st.markdown('### Analysing column relations')
st.write('Correlations:')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, ax=ax)
if st.checkbox('Show the Analysing column relations'):
    st.write(st.pyplot())

st.markdown('### Analysing the Target Variable')
st.text('visualizing Loan Status')
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='Loan Status',data=df,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
df['Loan Status'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
st.write(st.pyplot())
    


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

x= df.drop(columns=['Loan Status'])
y = df['Loan Status']

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)



alg = ['LogisticRegression', 'DecisionTreeClassifier']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='LogisticRegression':
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_lr = lr.predict(X_test)
    cm_lr=confusion_matrix(y_test,pred_lr)
    st.write('Confusion matrix: ', cm_lr)



elif classifier=='Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)


     
if classifier=='LogisticRegression':
    precision, recall, fscore, support = score(y_test, pred_lr)
    if st.checkbox('Show the Evaluation Metric'):
        st.text('precision: {}'.format(precision))
        st.text('recall: {}'.format(recall))
        st.text('fscore: {}'.format(fscore))
        st.text('support: {}'.format(support))


elif classifier=='LogisticRegression':
    precision, recall, fscore, support = score(y_test, pred_lr)
    if st.checkbox('Show the Evaluation Metric'):
        st.text('precision: {}'.format(precision))
        st.text('recall: {}'.format(recall))
        st.text('fscore: {}'.format(fscore))
        st.text('support: {}'.format(support))

if __name__ == '_main_':
    main()



  