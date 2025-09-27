#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")
plt.style.use("fivethirtyeight")
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor

from neuralprophet import NeuralProphet

import pmdarima as pm

from prophet import Prophet

from fbprophet import prophet
import lightgbm as lgb
from src.features import build_features
from src.data_prep import load_data
from pandas.tseries.holiday import USFederalHolidayCalender as calender


# In[50]:


file_path = r"C:\Users\koskr\Desktop\future_inturn\task_1\archive\Sample - Superstore.csv"


# In[51]:


pjme = pd.read_csv(file_path,
                  index_col=[0],
                  parse_dates=[0])
pjme.head()


# In[52]:


print(pjme.columns)


# In[53]:


pjme.rename(columns={'Order Date': 'ds'}, inplace=True)


# In[54]:


pjme.rename(columns={'Order Date': 'ds', 'pjme_MW': 'y'}, inplace=True)


# In[55]:


pjme['ds'] = pd.to_datetime(pjme['ds'])


# In[56]:


pjme = pjme.rename(columns={"Order Date ": "ds", "Sales": "y"})


# In[57]:


print(pjme.head())


# In[58]:


pjme["ds"] = pd.to_datetime(pjme["ds"])


# In[59]:


df['date'] = pd.to_datetime(df.index)


# In[60]:


df = pd.DataFrame()
cat_type = pd.DataFrame(['monday','tuesday',
                      'wednesday',"thursday",
                      'friday', "saturday",
                      "sunday"], 
                            )


# In[61]:


print(cat_type)


# In[62]:


from pandas.api.types import CategoricalDtype
cat_type = CategoricalDtype(
    categories=['monday','tuesday','wednesday','thursday',
                'friday','saturday','sunday'],
    ordered=True
)


# In[63]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# Categorical type for weekday
cat_type = CategoricalDtype(
    categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    ordered=True
)

def create_features(df, label=None):
    df = df.copy()
    
    # Make sure index is datetime
    df['date'] = pd.to_datetime(df.index)

    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    
    df['date_offset'] = (df['date'].dt.month*100 + df['date'].dt.day - 320) % 1300
    df['season'] = pd.cut(
        df['date_offset'], 
        [0,300,602,900,1300],
        labels=['spring', 'summer', 'fall', 'winter']
    )
    
    X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear','weekday','season']]
    
    if label:
        y = df[label]
        return X, y
    return X


# In[64]:


file_path_1 = r"C:\Users\koskr\Desktop\future_inturn\task_3\customer_support1\sample.csv"


# In[65]:


pjme_mw = pd.read_csv(file_path_1,
                  index_col=[0],
                  parse_dates=[0])
pjme_mw.head()


# In[66]:


X, y = create_features(pjme, label='y')  # <-- lowercase, doesn’t exist


# In[67]:


X, y = create_features(pjme, label='y')
features_and_target = pd.concat([X, y], axis=1)

fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(
    data=features_and_target.dropna(),
    x='weekday',
    y='y',
    hue='season',
    ax=ax,
    linewidth=1
)
ax.set_title('Power Use (MW) by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Energy (MW)')
ax.legend(bbox_to_anchor=(1,1))
plt.show()


# In[68]:


def create_features(df,label=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df.index)
    df['hour']=df['date'].dt
    df['dayofweek']=df['date'].dt
    df['weekofday'] = df['date'].dt.day_name()
    df['weekofday'] = df['weekofday'].astype(cat_type)
    df['quarters'] = df['date'].dt
    df['month']=df['date'].dt
    df['year']=df['date'].dt
    df['dayofyear']=df['date'].dt
    df['dayofmonth']=df['date'].dt
    df['weekofyear']=df['date'].dt
    df['date_offset']= (df.date.dt.month*100+df.date.dt.day-320)%1300
    df['season'] = pd.cut(df['date_offset'],[0,300,602,900,1300],
                     labels=['spring', "summer", 'fall', 'winter']
                     )
    X = df[['hour', 'dayofweek', 'quarters', 'month', 'year',
            'dayofyear','dayofmonth','weekofyear','weekofday',
            'season']]
    if label:
        y = df[label]
        return X, y
    return X


# In[69]:


print(df.head())


# In[70]:


X,y = create_features(pjme, label = 'y')


# In[71]:


fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(
    data=features_and_target.dropna(),
    x='weekday',
    y='y',
    hue='season',
    ax=ax,
    linewidth=1
)


# In[72]:


ax.set_title('Power Use (MW) by Day of Week')
ax.set_xlabel('Day of Week')
ax.set_ylabel('Energy (MW)')
ax.legend(bbox_to_anchor=(1,1))
plt.show()


# In[73]:


features_and_target.head() 


# In[74]:


split_date = pd.to_datetime("2015-01-01")


# In[75]:


pjme = pd.read_csv(file_path, parse_dates=['Order Date'], index_col='Order Date')


# In[ ]:





# In[76]:


print(type(pjme.index))
print(pjme.index[:5])


# In[77]:


print(pjme.columns)
print(pjme.head())


# In[78]:


pjme.index = pd.to_datetime(pjme.index)


# In[79]:


pjme.iloc[:,0] = pd.to_datetime(pjme.iloc[:,0])
pjme = pjme.set_index(pjme.columns[0])


# In[80]:


pjme = pjme.rename(columns={pjme.columns[0]: "y"})
pjme["ds"] = pjme.index


# In[81]:


split_date = pd.to_datetime("2015-01-01")
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test  = pjme.loc[pjme.index > split_date].copy()


# In[82]:


pjme_train_simple = pjme_train[['y']].rename(columns={'y': "Training Set"})
pjme_test_simple  = pjme_test[['y']].rename(columns={'y': "Test Set"})


# In[83]:


combined = pjme_test_simple.join(pjme_train_simple, how="outer")


# In[84]:


combined.plot(figsize=(15,5), title="PJME East", style=".")
plt.show()


# In[85]:


pjme_train_prophet = pjme_train.reset_index()    .rename(columns={'datetime' : 'ds',
                    'pjme mw' :'y'})
pjme_train_prophet.head()


# In[86]:


print(pjme['y'].dtype)
print(pjme['y'].head())


# In[87]:


pjme['y'] = pd.to_numeric(pjme['y'], errors='coerce')


# In[88]:


pjme = pjme.dropna(subset=['y'])  # or fillna(0) if appropriate


# In[89]:


pjme.index = pd.to_datetime(pjme.index)


# In[90]:


pjme['y'] = pjme['y'].interpolate()  # linear interpolation


# In[91]:


pjme['y'] = pjme['y'] / 1000  # optional scaling


# In[92]:


print(pjme['y'].dtype)


# In[93]:


print(pjme['y'].head())


# In[94]:


pjme['y'] = pjme['y'].astype(str).str.replace(',', '').astype(float)


# In[95]:


model = ARIMA(pjme['y'], order=(1,1,0))


# In[96]:


print(len(pjme['y']))


# In[97]:


model_fit = model.fit()


# In[98]:


forecast = model_fit.forecast(24)


# In[99]:


print(forecast)


# In[100]:


pjme.columns = [c.strip() for c in pjme.columns]  # remove extra spaces


# In[101]:


pjme['Datetime'] = pd.to_datetime(pjme.iloc[:, 0])  # first column is datetime


# In[102]:


pjme.set_index('Datetime', inplace=True)


# In[103]:


pjme['y'] = pd.to_numeric(pjme.iloc[:, 0 if pjme.shape[1]==1 else 1], errors='coerce')
pjme['y'] = pjme['y'].interpolate()  # fill missing values


# In[104]:


split_date = pd.to_datetime("2015-01-01")
pjme_train = pjme.loc[pjme.index <= split_date]
pjme_test  = pjme.loc[pjme.index > split_date]


# In[105]:


model = ARIMA(pjme_train['y'], order=(1,1,0))
model_fit = model.fit()


# In[106]:


forecast_steps = len(pjme_test)


# In[107]:


print(pjme_train.shape)  # number of rows in training
print(pjme_test.shape)   # number of rows in test
print(pjme_train['y'].dtype, pjme_test['y'].dtype)
print(pjme_train['y'].head())


# In[108]:


pjme['y'] = pd.to_numeric(pjme['y'], errors='coerce')
pjme['y'] = pjme['y'].interpolate()
pjme = pjme.dropna(subset=['y'])


# In[109]:


forecast_steps = min(len(pjme_test), 20)  # do not exceed test length


# In[110]:


plt.figure(figsize=(15,5))


# In[111]:


plt.plot(pjme_train['y'], label='Train')


# In[112]:


plt.plot(pjme_test['y'], label='Test')


# In[113]:


print(pjme_train.shape)
print(pjme_train['y'].dtype)
print(pjme_train['y'].isna().sum())


# In[114]:


pjme['y'] = pd.to_numeric(pjme['y'], errors='coerce')
pjme['y'] = pjme['y'].interpolate()  # fill missing values
pjme = pjme.dropna(subset=['y'])


# In[115]:


model = ARIMA(pjme_train['y'], order=(1,1,0))
model_fit = model.fit()


# In[116]:


plt.figure(figsize=(15,5))


# In[117]:


plt.plot(pjme_train['y'], label='Train')


# In[118]:


plt.plot(pjme_test['y'], label='Test')


# In[119]:


plt.figure(figsize=(15,5))
plt.plot(pjme_train['y'], label='Train')
plt.plot(pjme_test['y'], label='Test')
plt.xlabel("Datetime")
plt.ylabel("MW")
plt.title("PJME Forecast Using ARIMA")
plt.legend()
plt.show()


# In[120]:


holidays = pd.DataFrame({
  'holiday': 'Diwali',
  'ds': pd.to_datetime(['2025-10-20', '2026-11-08']),
  'lower_window': 0,
  'upper_window': 1,
})


# In[121]:


forecast_steps = 24
pjme_forecast = model_fit.forecast(steps=forecast_steps)


# In[122]:


print(pjme.head())
print(pjme.shape)


# In[123]:


forecast_steps = 24
pjme_forecast = model_fit.forecast(steps=forecast_steps)

# Only if pjme is not empty
if not pjme.empty:
    last_date = pjme.index.max()  # safe alternative to index[-1]
    forecast_index = pd.date_range(last_date, periods=forecast_steps+1, freq="H")[1:]

    forecast_df = pd.DataFrame({
        "ds": forecast_index,
        "Forecast": pjme_forecast
    })

    plt.figure(figsize=(12,5))
    plt.plot(pjme['y'], label="History")
    plt.plot(forecast_df['ds'], forecast_df['Forecast'], label="Forecast", linestyle="--")
    plt.legend()
    plt.show()
else:
    print("⚠️ pjme DataFrame is empty — check your data loading/renaming.")


# In[124]:


print(pjme.columns)
print(pjme.head())


# In[125]:


pjme = pjme.rename(columns={
    'Datetime': 'ds',
    'PJME MW': 'y'
})
...


# In[126]:


print(pjme.columns.tolist())


# In[127]:


pjme = pjme.rename(columns={
    'Datetime': 'ds',
    'PJME MW': 'y'
})


# In[128]:


pjme = pjme.rename(columns={
    'datetime': 'ds',
    'pjme mw': 'y'
})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[129]:


split_date = "2015-01-01"
train = pjme.loc[pjme.index < split_date]
test = pjme.loc[pjme.index >= split_date]


# In[130]:


pjme['ds'] = pd.to_datetime(pjme['ds'])
pjme = pjme.set_index('ds')
pjme = pjme.asfreq('H')  # hourly frequency
pjme['y'] = pd.to_numeric(pjme['y'], errors='coerce').interpolate()


# In[131]:


pjme.columns = pjme.columns.str.strip()  # remove spaces
pjme = pjme.rename(columns={'PJME MW': 'y'})  # or your actual column name
pjme['y'] = pd.to_numeric(pjme['y'], errors='coerce').interpolate()
pjme = pjme.dropna(subset=['y'])


# In[132]:


color_pal = sns.color_palette()
pjme.plot(style='.', figsize=(10,5), ms=1, color=color_pal[0], title="PJME MW")
plt.show()


# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[133]:


import pandas as pd
df = pd.DataFrame()
cat_type = pd.DataFrame(['monday','tuesday',
                      'wednesday',"thursday",
                      'friday', "saturday",
                      "sunday"], 
                            )
print(cat_type)


# In[ ]:





# In[134]:


pjme_train_plot = pjme_train[['y']]   # keep only target
pjme_test_plot = pjme_test[['y']]

forecast_plot = pjme_test_plot.rename(columns={'y': 'Test Set'})     .join(
        pjme_train_plot.rename(columns={'y': 'Training Set'}),
        how='outer'
    )

forecast_plot.plot(figsize=(15,5), title="PJME East", style='.')
plt.show()


# In[135]:


forecast_plot = pjme_test.rename(columns={'y': 'Test Set'})     .join(
        pjme_train.rename(columns={'y': 'Training Set'}),
        lsuffix='_test', rsuffix='_train',
        how='outer'
    )


# In[ ]:





# In[136]:


pjme_train_prophet = pjme_train.reset_index()    .rename(columns={'datetime' : 'ds',
                    'pjme mw' :'y'})
pjme_train_prophet.head()


# In[ ]:





# In[137]:


split_date = "2015-01-01"
train = pjme.loc[pjme.index < split_date]
test = pjme.loc[pjme.index >= split_date]


# In[138]:


train['y'] = pd.to_numeric(train['y'], errors='coerce')
train = train.dropna()


# In[139]:


model = SARIMAX(train['y'], order=(1,1,1), seasonal_order=(0,1,1,24))


# In[140]:


model = SARIMAX(train['y'], order=(1,1,1))


# In[141]:


train = pjme_train_prophet.set_index("ds")["y"]

# Fit SARIMAX (ARIMA with daily seasonality = 24 steps if hourly data)
model = SARIMAX(train,
                order=(1,1,1),
                seasonal_order=(1,1,1,24),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

print(model_fit.summary())


# In[142]:


pjme = pd.concat([pjme_train, pjme_test])


# In[143]:


print(pjme.columns.tolist())


# In[144]:


pjme = pjme.rename(columns={
    'Datetime': 'datetime',  # or 'datetime' if lowercase
    'PJME MW': 'y'           # your target column
})


# In[145]:


pjme.columns = pjme.columns.str.strip()
pjme = pjme.rename(columns={
    'Datetime': 'datetime',
    'PJME MW': 'y'
})


# In[146]:


print(pjme.columns.tolist())


# In[147]:


pjme = pjme.rename(columns={'Datetime': 'datetime'})


# In[148]:


pjme.columns = pjme.columns.str.strip()
pjme = pjme.rename(columns={'Datetime': 'datetime'})


# In[ ]:





# In[149]:


pjme.columns = pjme.columns.str.strip()  # remove leading/trailing spaces
pjme = pjme.rename(columns={'Datetime': 'datetime', 'PJME MW': 'y'})


# In[150]:


pjme_train_prophet['hour'] = pjme_train_prophet['ds'].dt.hour


# In[151]:


pjme_train_prophet['dayofweek'] = pjme_train_prophet['ds'].dt.dayofweek


# In[152]:


pjme_train_prophet['month'] = pjme_train_prophet['ds'].dt.month


# In[153]:



X = pjme_train_prophet[['hour','dayofweek','month']]
y = pjme_train_prophet['y']


# In[154]:


model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[155]:


model.fit(X, y)


# In[156]:



print(pjme.columns.tolist())
print(pjme.head())
print(pjme.shape) 


# In[157]:



model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[158]:


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred  = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100


# In[ ]:





# In[159]:


fig, ax = plt.subplots(figsize=(10,5))


# In[160]:


plt.figure(figsize=(15,5))


# In[161]:


train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

train = train.rename(columns={'Datetime': 'datetime'})  # adjust to match your column
test = test.rename(columns={'Datetime': 'datetime'})


# In[162]:


plt.xlabel("Datetime")
plt.ylabel("MW")
plt.title("PJME Forecast with Random Forest")
plt.legend()
plt.show()


# In[163]:


fig = model.plot_components(pjme_test_fcst)
plt.show()


# In[164]:



model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
forecast = model.predict(X_test)


# In[165]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

plt.plot(train['datetime'], y_train, label='Train')
plt.plot(test['datetime'], y_test, label='Test')
plt.plot(test['datetime'], forecast, label='Forecast', linestyle='--')

plt.xlabel("Datetime")
plt.ylabel("MW")
plt.title("PJME Forecast")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




