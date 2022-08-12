import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'F:\books\پروژه\bupa.data',header= None)

print(df.columns.name)
print(df.index)
print(df.columns)
df.columns= ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks','t']
print(df.columns)
print(df)

print(df.isnull().sum())

#plt.boxplot(df)
#plt.show()

print(df.value_counts('mcv', normalize=True, sort= True))


plt.scatter(x=df.mcv, y= df.drinks)
plt.xlabel('mean_corpuscular_volume')
plt.ylabel('drinks')
#plt.show()

#sns.heatmap(df[['mcv', 'sgpt', 'drinks']].corr(method= ))
#plt.show()

#test= df.loc['t']

#train= np.empty([200, 7])
#test= np.empty([144, 7])
Counter= 0
counter= 0
df_train= pd.DataFrame
dff_train= pd.DataFrame
for i in range(344):

    if df.at[i, 't']==1:
        counter += 1
        #test= np.array(df.iloc[i])
    elif df.at[i, 't']==2:

        dff_train = df_train
        df_train= pd.DataFrame(df.iloc[[i]])
        #frames= [dff_train, df_train]
        #pd.concat(frames)

        df_train= pd.concat([dff_train, df_train], axis=0)

        Counter += 1
        #print( df_train)
        print(dff_train)
print(counter)
print(Counter)
        #print(test)

# print(df.at[0, 't'])

df= df._drop_axis('t', axis=1)
w0= pd.DataFrame(np.ones((345, 1)))
#print(w0)
df_new= pd.concat([w0, df], axis=1)
print(df_new)

y= df.xs('drinks', axis=1)
print(y)

#linear_regressor = LinearRegression().fit()
XTX= df_new.T.dot(df_new)
XTX_inv= np.linalg.inv(XTX)
##f= XTX.dot(XTX_inv).round(1)
##print(f)
w= XTX_inv.dot(df_new.T).dot(y)
print(w)