import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.linear_model import LinearRegression
import time

t0 = time.time()
################

r = 1/1.06
K = 1.1

data = np.array([[1, 1.09, 1.08, 1.34], [1, 1.16, 1.26, 1.54], [1, 1.22, 1.07, 1.03],
             [1, 0.93, 0.97, 0.92], [1, 1.11, 1.56, 1.52], [1, 0.76, 0.77, 0.90],
             [1, 0.92, 0.84, 1.01], [1, 0.88, 1.22, 1.34]])

SPP = pd.DataFrame(data, columns = ['X_0', 'X_1', 'X_2', 'X_3'])

m = len(data[0])-1

cfm = np.maximum(data-1.1, 0)

print(cfm)



for i in range(len(data[0])-1, 0, -1):
    print(data[:,i]**2)

for i in range(1,m+1):
    SPP[f'X_{i}^2'] = SPP[f'X_{i}'] ** 2

SPP['id'] = SPP.index

# Cash flow matrix
df = pd.DataFrame([])
df['id'] = SPP['id']

# DataFrame for Regression
for i in range(m,0,-1):
    df[f'CFM_{i}'] = np.maximum(K - SPP[f'X_{i}'], 0)
    #print(df[f'CFM_{i}'])

print()

for i in range(m-1, 0, -1):

    df[f'bcV_{i}'] = df[f'CFM_{i + 1}'] * r

    valid_ids = df.loc[df[f'CFM_{i}'] > 0, 'id']
    filtered_SPP = SPP[SPP['id'].isin(valid_ids)]


    X = filtered_SPP[[f'X_{i}', f'X_{i}^2']]
    Y = df.loc[df['id'].isin(valid_ids), f'bcV_{i}']

    model = LinearRegression()
    model.fit(X, Y)

    a = model.intercept_
    b, c = model.coef_
    #test for at se om de rigtige tal bliver brugt
    #print(f'Iteration {i}: a={a}, b={b}, c={c}')
    #print(f'Iteration {i}:')
    #print(f'X:\n{X.head()}')
    #print(f'Y:\n{Y.head()}')

    E = a + b * filtered_SPP[f'X_{i}'] + c * filtered_SPP[f'X_{i}^2']
    df.loc[df['id'].isin(valid_ids), f'E[X_{i}|Y]'] = E.values


    # this is to override the CFM (if we continue with option in future, we value at 0)
    valid_ids_1 = df.loc[df[f'E[X_{i}|Y]'] > df[f'CFM_{i}'], 'id']
    filtered_df = df[df['id'].isin(valid_ids_1)]

    new_value = 0
    df.loc[df['id'].isin(valid_ids_1), f'CFM_{i}'] = new_value

    # this is to override the CFM at the later time, if we have a positiv CFM in a later period
    valid_ids_2 = df.loc[df[f'CFM_{i}'] > 0, 'id']
    filtered_df = df[df['id'].isin(valid_ids_2)]

    new_value = 0
    df.loc[df['id'].isin(valid_ids_2), f'CFM_{i + 1}'] = new_value


#Now i want to define a new dataframe that only selects columns that include cash flows
selected_columns = []
for col in df.columns:
    if col.startswith('CFM'):
        selected_columns.append(col)

Cash_Flow_Matrix = df[selected_columns]
Cash_Flow_Matrix = Cash_Flow_Matrix.iloc[:, ::-1]

print(Cash_Flow_Matrix)

# Finding the back discounted value
p = []
for i in range(1,m):
    x = r**(i) * sum(Cash_Flow_Matrix[f'CFM_{i}']) / len(Cash_Flow_Matrix)
    p.append(x)

Price_US = sum(p)

Price_EU = r**3 * sum(np.maximum(K - SPP['X_3'], 0)) / len(SPP)


print(Price_US, Price_EU)
print(sum(Cash_Flow_Matrix['CFM_1']))


################
t1 = time.time()
total = t1-t0
print('time:',total)