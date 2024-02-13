#!/usr/bin/env python
# coding: utf-8

# In[101]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # Introduction 

# The aim is to predict sales for various product families (i.e. Automotive, Babycare, Beauty, Books etc.) sold at Favorita stores located in Ecuador. The training data includes dates, store number, product information, whether the product was on promotion, and sales numbers. 
# 
# Supplementary information is provided in other files which can be used to help build the model.

# #### Additional Notes
# 
# Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.
# A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.

# ## Step 1 - Preprocess datasets (training data and supplementary data)

# ### Clean Data

# In[102]:


#import libraries

import pandas as pd

import matplotlib.pyplot as plt

import plotly as py
import plotly.express as px

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns



import pandas as pd
import numpy as np


# In[103]:


train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\train.csv")
train_data.head()


# In[104]:


train_data.columns


# In[105]:


train_data.shape


# In[106]:


train_data.describe()


# In[107]:


train_data.info()


# In[108]:


print('Missing values (%) per column: \n', 100*train_data.isnull().mean())


# In[109]:


#let's take a look at the supplementary datasets
hol_events = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\holidays_events.csv")
hol_events.head()


# In[139]:


hol_events['type'].isna().sum()


# In[110]:


missing_hol_vals = hol_events.isna().sum()
missing_hol_vals


# In[111]:


oil_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\oil.csv")
oil_df.head()


# In[112]:


missing_oil_vals = oil_df.isna().sum()
missing_oil_vals


# In[113]:


oil_df.describe()


# In[114]:


#determine median oil price

import math 
median_oil_price = math.floor(oil_df.dcoilwtico.median())
median_oil_price


# In[115]:


#let's fill in the missing values in the oil price column with the median value as this is more robust to outliers  

oil_df.dcoilwtico = oil_df.dcoilwtico.fillna(median_oil_price)
oil_df.head()


# In[116]:


stores_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\stores.csv")
stores_df.head()


# In[117]:


missing_store_vals = stores_df.isna().sum()
missing_store_vals


# In[118]:


transactions_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\transactions.csv")
transactions_df.head()


# In[119]:


missing_trans_vals = transactions_df.isna().sum()
missing_trans_vals


# ### Reshape Data

# In[120]:


#re-shape data to merge all datasets on date and store number
#join matching rows form supplementary data to training data
merged_train_df = pd.merge(train_data, hol_events, how = 'left', on = 'date')
merged_train_df.head()


# In[121]:


merged_train_df = pd.merge(merged_train_df, oil_df, how = 'left', on = 'date')
merged_train_df.head()


# In[122]:


merged_train_df = pd.merge(merged_train_df, stores_df, how = 'left', on = 'store_nbr')
merged_train_df.head()


# In[123]:


merged_train_df = pd.merge(merged_train_df, transactions_df, how = 'left', on = ['date', 'store_nbr'])
merged_train_df.head()


# In[124]:


#let's split the date column into month and year

merged_train_df['date'] = pd.to_datetime(merged_train_df['date'])
#data['date'] = pd.to_datetime(data['year'], format='%Y').dt.year

merged_train_df['Day'] = merged_train_df['date'].dt.day_name()
merged_train_df['Month'] = pd.to_datetime(merged_train_df['date'], format = '%m').dt.month
merged_train_df['Year'] = pd.to_datetime(merged_train_df['date'], format='%Y').dt.year


# In[125]:


merged_train_df.head()


# In[140]:


merged_train_df.isna().sum()


# In[141]:


print('Missing values (%) per column: \n', 100*merged_train_df.isnull().mean())


# In[126]:


merged_train_df['type_x'].value_counts()


# The column 'type_x' indicates if a given date is a holiday or not, and if it is a holiday there may be some adjustments made. Please note the following adjustments for a given category:- 
# 
# - Holiday :  a holiday 
# - Event : an event 
# - Additional : Extra days added to a regular calendar holiday such as Christmas 
# - Transfer : the new date an official "holiday" has been transferred to
# - Work Day : Makes up for the dates taken as "bridge" days
# - Bridge : Extra days added to a "holiday" to extend the break over a long weekend
#     

# Key things to bear in mind during the analysis: - 
# - public sector wages are paid every 2 weeks on the 15th and on the last day of the month
# - a 7.8 magnitude earthquake struck Ecuador on the 16th of April 2016. Aid was donated by others around this time which greatly affected supermarket sales

# ## Data Analysis 

# In[127]:


merged_train_df.groupby('Year')['sales'].count()


# In[128]:


fig = px.histogram(merged_train_df, x = 'Year', title = 'Sales in each city by year', color = 'city')
fig.update_layout(bargap=0.2)
fig.show()


# In[129]:


sns.lineplot(x='store_nbr', y = 'sales', data =merged_train_df)


# In[30]:


sns.lineplot(x='Year', y = 'sales', data =merged_train_df)


# In[31]:


sns.lineplot(x='Year', y = 'dcoilwtico', data =merged_train_df)


# In[32]:


sales=merged_train_df.pivot_table(index="Day", columns="Month", values='sales', aggfunc='mean')
sales


# In[63]:


mean_sales = merged_train_df['sales'].mean().round(2)
mean_sales


# In[33]:


plt.figure(figsize=(12,6))
sns.heatmap(sales, cmap='Blues', annot=True, fmt='.1f')
plt.title('Average sales by Month and Day of the Week')
plt.show()


# From the heatmap above, it is clear that sales are above average (359.02) on the weekends. Also, sales hit a peak in the month of November, with October and December also performing well for sales. 

# In[34]:


merged_train_df.head()


# In[35]:


plt.plot(merged_train_df['date'], merged_train_df['transactions'])
plt.title('Transactions 2013 - 2017')
plt.legend(['transactions'])
plt.show()


# In[36]:


#lets take a closer look at transactions by store number
store_figs_df = merged_train_df.groupby('store_nbr')['transactions'].sum().sort_values(ascending=False).reset_index()
store_figs_df


# In[37]:


fig = px.bar(store_figs_df, x = 'store_nbr', y = 'transactions', title = 'Total number of transactions by store number, 2013 - 2017')
fig.show()


# In[61]:


#lets take a closer look at transactions by store number
store_sales_df = merged_train_df.groupby(['store_nbr', 'Year', 'family'])['sales'].sum().round(2).reset_index()
store_sales_df


# In[ ]:





# In[38]:


merged_train_df.head()


# In[39]:


fig = px.line(merged_train_df, x ='date', y ='transactions', color='store_nbr', title='Transactions by store number, 2013-2017')
fig.show()


# In[40]:


#lets take a closer look at transactions by product family 
prod_fam_figs_df = merged_train_df.groupby('family')['transactions'].sum().sort_values(ascending=False).reset_index()
prod_fam_figs_df


# Let's ask ourselves what factors affect sales numbers? 
# 
# - promotions
# - holiday/events
# - state of domestic economy
# - state of world economy 
# - domestic pay 
# - shopping trends
# - location/ geography
# 

# In[65]:


merged_train_df.columns


# In[71]:


#let's take a closer look at promo prod.family, quantities, dates and sales numbers 

small_df = merged_train_df.groupby(['Year', 'family', 'onpromotion'])['sales'].sum().round(2).reset_index()


# In[72]:


small_df


# In[82]:


merged_train_df.dtypes


# In[89]:


num_merged_df = merged_train_df.select_dtypes(include='number')
num_merged_df


# In[90]:


num_merged_df.corr().style.background_gradient(cmap='Oranges')


# r-values greater than 0.7 indicate a strong correlation between two features. From the graph above, there seems to be a strong correlation between the year and the product id (perhaps, as the year increases, so does the id number). Other features which show signs of a correlation are promotional items and sales (0.4). 
# 

# In[130]:


#let's assign numerical values to some of the categorical data to carry out more in-depth analysis (i.e. family, )

merged_train_df['family'].unique()


# In[131]:


fam_d={}
for index, key in enumerate(merged_train_df['family'].unique(), start=1):
    fam_d[key] = index

print(fam_d)


# In[132]:


merged_train_df['family'] = merged_train_df['family'].map(fam_d)
merged_train_df.head()


# In[133]:


merged_train_df['family'].unique()


# In[135]:


merged_train_df['type_x'].unique()


# In[136]:


hol_d={}
for index, key in enumerate(merged_train_df['type_x'].unique(), start=1):
    hol_d[key] = index

print(hol_d)


# In[138]:


merged_train_df.loc[merged_train_df['type_x'].isna()]


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


#Let's take a closer look at the spread of the sales data 
plt.figure(figsize = (12,8))

sns.boxplot(data=train_data, y ='sales', showmeans=True)

mean_sales=train_data['sales'].mean()
median_sales=train_data['sales'].median()
plt.axhline(y=mean_sales, color='r', linestyle='-')
plt.axhline(y=median_sales, color='g', linestyle='-')

plt.title('Sales Data')

plt.show()


# In[42]:


train_data.groupby('family')['sales'].sum()


# In[43]:


train_data['family'].nunique()


# In[44]:


train_data['sales'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




