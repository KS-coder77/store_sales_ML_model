#!/usr/bin/env python
# coding: utf-8

# In[121]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# #### Define The Problem ...
# 
# Predict sales over a period of time for thousands of product families sold in a chain of shops in Ecuador. 
# 
# #### Define the Type of Problem ...
# 
# We can use regression analysis based on past observations to predict future sales figures (i.e. continuous numerical values based on input features).   
# 

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

# In[122]:


#import libraries
import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly as py
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objs as go

sns.set_theme(style='darkgrid')


# In[123]:


train_data = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\train.csv")
train_data.head()


# In[124]:


train_data.columns


# File Descriptions and Data Field Information
# 
# train.csv
# - The training data, comprising time series of features 
# * store_nbr: identifies the store at which the products are sold.
# * family: identifies the type of product sold.
# * onpromotion: gives the total number of items in a product family that were being promoted at a store at a given date.
# * sales: gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
# 
# test.csv
# - The test data, having the same features as the training data. You will predict the target sales for the dates in this file.
# - The dates in the test data are for the 15 days after the last date in the training data.
# 
# sample_submission.csv
# - A sample submission file in the correct format 
# 
# stores.csv
# - Store metadata, including city, state, type, and cluster.
# - cluster is a grouping of similar stores.
# 
# - oil.csv
# Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
# 
# - holidays_events.csv
# Holidays and Events, with metadata
# 
# NOTE: Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type Bridge are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type Work Day which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
# Additional holidays are days added a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

# In[125]:


train_data.shape


# In[126]:


train_data.describe()


# In[127]:


train_data.info()


# In[128]:


train_data['date'] = pd.to_datetime(train_data.date)


# In[129]:


print('Missing values (%) per column: \n', 100*train_data.isnull().mean())


# In[130]:


#let's take a look at the supplementary datasets
hol_events = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\holidays_events.csv")
hol_events.head()


# In[131]:


hol_events['locale_name'].nunique()


# In[132]:


hol_events['date'] = pd.to_datetime(hol_events.date)


# In[133]:


hol_events['type'].isna().sum()


# In[134]:


missing_hol_vals = hol_events.isna().sum()
missing_hol_vals


# In[135]:


oil_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\oil.csv")
oil_df.head()


# In[136]:


oil_df['date']= pd.to_datetime(oil_df.date)


# In[137]:


missing_oil_vals = oil_df.isna().sum()
missing_oil_vals


# In[138]:


oil_df.describe()


# In[139]:


#let's fill the missing values in the oil price column using interpolation 
oil_df["dcoilwtico"] = np.where(oil_df["dcoilwtico"] == 0, np.nan, oil_df["dcoilwtico"])
oil_df["dcoilwtico_interpolated"] =oil_df.dcoilwtico.interpolate()


# In[140]:


values = oil_df['dcoilwtico'].values
values


# In[141]:


indices_nan = np.isnan(values)
indices_nan


# In[142]:


# Get indices where values are not NaN
indices_not_nan = ~indices_nan

# Linearly interpolate NaN values
interpolated_values = np.interp(np.arange(len(values)), 
                                 indices_not_nan.nonzero()[0], 
                                 values[indices_not_nan])

# Replace NaN values with interpolated values
values[indices_nan] = interpolated_values[indices_nan]

# Update the DataFrame with the interpolated column
oil_df['dcoilwtico'] = values


# In[143]:


oil_df.head(25)


# In[144]:


oil_df.isna().sum()


# In[145]:


stores_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\stores.csv")
stores_df.head()


# In[146]:


stores_df['state'].nunique()


# In[147]:


missing_store_vals = stores_df.isna().sum()
missing_store_vals


# In[148]:


transactions_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\transactions.csv")
transactions_df.head()


# In[149]:


transactions_df.info()


# In[150]:


transactions_df['date'] =  pd.to_datetime(transactions_df.date)


# In[151]:


missing_trans_vals = transactions_df.isna().sum()
missing_trans_vals


# ### Reshape Data

# In[193]:


#re-shape data to merge all datasets on date and store number
#join matching rows form supplementary data to training data
merged_train_df = pd.merge(train_data, hol_events, how = 'left', on = 'date')
merged_train_df = pd.merge(merged_train_df, oil_df, how = 'left', on = 'date')
merged_train_df = pd.merge(merged_train_df, stores_df, how = 'left', on = 'store_nbr')
merged_train_df = pd.merge(merged_train_df, transactions_df, how = 'left', on = ['date', 'store_nbr'])

merged_train_df.head()


# In[194]:


#let's split the date column into month and year

merged_train_df['date'] = pd.to_datetime(merged_train_df['date'])
#data['date'] = pd.to_datetime(data['year'], format='%Y').dt.year

merged_train_df['Day'] = merged_train_df['date'].dt.day_name()
merged_train_df['Month'] = pd.to_datetime(merged_train_df['date'], format = '%m').dt.month
merged_train_df['Year'] = pd.to_datetime(merged_train_df['date'], format='%Y').dt.year


# In[195]:


merged_train_df.head()


# In[196]:


print('Missing values (%) per column: \n', 100*merged_train_df.isnull().mean())


# In[197]:


#let's fill the missing categorical data with the mode value

mode_hol = merged_train_df['type_x'].mode().iloc[0]
mode_loc = merged_train_df['locale'].mode().iloc[0]
mode_loc_name = merged_train_df['locale_name'].mode().iloc[0]
mode_descp = merged_train_df['description'].mode().iloc[0]
mode_transf = merged_train_df['transferred'].mode().iloc[0]

merged_train_df['type_x'].fillna(mode_hol, inplace=True)
merged_train_df['locale'].fillna(mode_loc, inplace=True)
merged_train_df['locale_name'].fillna(mode_loc_name, inplace=True)
merged_train_df['description'].fillna(mode_descp, inplace=True)
merged_train_df['transferred'].fillna(mode_transf, inplace=True)


# In[198]:


#let's fill the missing numerical data with the median value /interpolated values
median_trans = merged_train_df['transactions'].median()

merged_train_df['transactions'].fillna(median_trans, inplace=True)
merged_train_df["dcoilwtico"] = np.where(merged_train_df["dcoilwtico"] == 0, np.nan, merged_train_df["dcoilwtico"])
merged_train_df["dcoilwtico_interpolated"] = merged_train_df.dcoilwtico.interpolate()
merged_train_df["dcoilwtico"] = merged_train_df["dcoilwtico_interpolated"]
merged_train_df = merged_train_df.drop(columns='dcoilwtico_interpolated')
merged_train_df.head()


# In[199]:


merged_train_df.isnull().sum()


# Now the datasets have been merged into one and the missing values have been dealt with, let's dive into the data analysis! 

# In[200]:


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

# In[201]:


merged_train_df.shape


# In[202]:


train_data.shape


# ## Data Analysis 

# ### 1. Train data

# In[120]:


train_data.head()


# some stores have zero sales for certain product families (i.e. store number 1 has zero automotive sales)

# In[44]:


#let's split the date column into day/month/year

train_data['date'] = pd.to_datetime(train_data['date'])

train_data['Day'] = train_data['date'].dt.day_name()
train_data['Month'] = pd.to_datetime(train_data['date'], format = '%m').dt.month
train_data['Year'] = pd.to_datetime(train_data['date'], format='%Y').dt.year


# In[45]:


train_data.head()


# In[46]:


#sales by year 
fig = px.histogram(train_data, x='Year', title = "Sales by Year", color= train_data['family'])
fig.show()


# In[47]:


sales_df = train_data.pivot_table(index="Day", columns="Month", values='sales', aggfunc='mean')
sales_df


# In[48]:


plt.figure(figsize=(12,6))
sns.heatmap(sales_df, cmap='Blues', annot=True, fmt='.1f')
plt.title('Average sales by Month and Day of the Week')
plt.show()


# From the heatmap above, it is clear that sales are above average (359.02) on the weekends. Also, sales hit a peak in the month of November, with October and December also performing well for sales. 

# In[49]:


color= train_data['sales']


# ### What is the difference between sales and transactions ?
# 
# - Sales gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips)
# 
# 
# - Transactions gives the total transactions at a particular store at a given date.

# In[220]:


#daily sales by store_nbr
sales_by_store = train_data.set_index('date').groupby('store_nbr').resample('D').sales.sum().reset_index()
px.line(sales_by_store, x='date', y='sales', color='store_nbr', title='Daily Total Sales by Store Number')


# The graph above illustrates when peak sales occurred between 2013 and 2017. Namely, Store Number 2 had an amazing day of sales on May 2nd 2016.

# In[51]:


sales_by_store.sort_values('sales', ascending=False)


# From the table, we can deduce that 2016 was a bumper year for sales, whereas 2013 was a slower year. 

# In[52]:


#let's visualise the top performing stores in another way

store_sales_df = train_data.groupby('store_nbr').sales.mean().sort_values(ascending=False).reset_index()
store_sales_df


# In[251]:


store_bar_sales = px.bar(store_sales_df, y='sales', x='store_nbr', color='sales', title='Store performance')
store_bar_sales


# It is clear from the graph above that store Number 44 has the highest number of sales overall, followed by store numbers 45, 47, 3 and 49. Whereas, stores 32, 22 and 52 are lowest performing. 

# In[54]:


sales_by_family = train_data.groupby('family').sales.mean().sort_values(ascending=False).head(10).reset_index()
sales_by_family


# In[55]:


px.bar(sales_by_family, y='family', x='sales', color='family', title='Top 10 Sales by Product Families ')


# Overall, the top product family is Grocery I, followed by Beverages, Produce, Cleaning, Dairy. 

# ### 2. Holidays and Events dataset

# In[56]:


hol_events.head()


# In[57]:


fig = px.histogram(hol_events, x='type', title = "Holiday Type Transferred Histogram", color= hol_events['transferred'])
fig.show()


# The graph above summarises the following across 2012 - 2017: - 
# - 12 "holidays" were transferred (i.e. a holiday was moved to another date by the government)
# - 2 "bridge" days were added (i.e. extra days added to a holiday to extend the break across a long weekend)
# - 51 "additional" days were added to the calendar for typical holidays around Christmas for example
# - 5 "work days" a day not normally scheduled for work (e.g. Saturday) that is meant to payback the Bridge
# - 56 "events"
# 

# In[58]:


#let's see the split by year group

hol_events['date'] = pd.to_datetime(hol_events['date'])

hol_events['Day'] = hol_events['date'].dt.day_name()
hol_events['Month'] = pd.to_datetime(hol_events['date'], format = '%m').dt.month
hol_events['Year'] = pd.to_datetime(hol_events['date'], format='%Y').dt.year


# In[59]:


hol_events.head()


# In[60]:


fig = px.histogram(hol_events, x='Year', title = "Holiday Type by Year", color= hol_events['type'], barmode='group')
fig.show()


# ### 3. Oil Price dataset

# In[61]:


oil_df.head()


# The oil dataset provides daily oil prices from 1st Jan 2013 to 31st August 2017.

# In[62]:


oil = oil_df.set_index('date').dcoilwtico.resample('D').sum().reset_index()
oil.head()


# In[63]:


oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()


# In[64]:


oil.head(25)


# In[65]:


oil_int_df = oil.melt(id_vars=['date']+list(oil.keys()[5:]), var_name='Legend')
px.line(oil_int_df.sort_values(["Legend", "date"], ascending = [False, True]), x='date', y='value', color='Legend',title = "Daily Oil Price" )


# From the graph above, oil prices hit a peak in September 2013, and steadily fell from September 2014 to Jan 2015. The lowest period was Jan 2016. The earthquake in April 2016 did not have a big impact on oil prices.

# In[66]:


#let's replace the NAN values with the interpolated values
values = oil['dcoilwtico'].values
values


# In[67]:


indices_nan = np.isnan(values)
indices_nan


# In[68]:


# Get indices where values are not NaN
indices_not_nan = ~indices_nan

# Linearly interpolate NaN values
interpolated_values = np.interp(np.arange(len(values)), 
                                 indices_not_nan.nonzero()[0], 
                                 values[indices_not_nan])

# Replace NaN values with interpolated values
values[indices_nan] = interpolated_values[indices_nan]

# Update the DataFrame with the interpolated column
oil['dcoilwtico'] = values


# In[69]:


oil = oil.drop(columns = 'dcoilwtico_interpolated')
oil.head(25)


# Let's see if there is any correlation between oil prices and sales/transactions. History suggests that when oil prices are high, the economy is struggling and product prices are expected to be high and sales therefore low.

# In[70]:


#merge sales data with oil data
sales_oil_df = pd.merge(sales_by_store, oil, how='left')
sales_oil_df


# In[71]:


px.scatter(sales_oil_df, x = "dcoilwtico", y = "sales", trendline = "ols", trendline_color_override = "red")


# Given the regression line is almost horizontal (i.e. flat), this indicates there is little to no relationship between oil price and the number of sales.
# 

# ### 4. Store dataset

# In[72]:


stores_df.head()


# In[73]:


stores_df['state'].nunique()


# In[77]:


fig = px.histogram(stores_df, x='type', title = "Store Type by City Histogram", color= stores_df['city'])
fig.show()


# The majority of stores are type D, followed by type C. With the majority of them located in the capital city, Quito. 

# In[76]:


fig = px.histogram(stores_df, x='type', title = "Store Type by State Histogram", color= stores_df['state'])
fig.show()


# ### 5. Transactions dataset

# In[182]:


transactions_df.head()


# In[183]:


transactions_df['date'] = pd.to_datetime(transactions_df['date'])

transactions_df['Day'] = transactions_df['date'].dt.day_name()
transactions_df['Month'] = pd.to_datetime(transactions_df['date'], format = '%m').dt.month
transactions_df['Year'] = pd.to_datetime(transactions_df['date'], format='%Y').dt.year


# In[184]:


transactions_df.head()


# In[185]:


#transactions by year
transactions_df.groupby('Year')['transactions'].sum().to_frame()


# In[186]:


avg_trans = transactions_df.pivot_table(index="Day", columns="Month", values='transactions', aggfunc='mean')
avg_trans


# In[187]:


plt.figure(figsize=(12,6))
sns.heatmap(avg_trans, cmap='Reds', annot=True, fmt='.1f')
plt.title('Average Transactions by Month and Day of the Week')
plt.show()


# Like Sales figures, transactions are very high on Sundays and in the months October to December. There seems to be a linear relationship between sales and transactions. 

# In[85]:


#daily transactions by store_nbr

trans_df = transactions_df.set_index('date').groupby('store_nbr').resample('D').transactions.sum().reset_index()
trans_df.head()


# In[86]:


px.line(trans_df, x='date', y='transactions', color='store_nbr',title = "Daily Total Transactions by Store Number" )


# In[252]:


store_bar_sales


# In[263]:


store_trans_df = trans_df.groupby('store_nbr').transactions.mean().sort_values(ascending=False).reset_index()
store_sales_df
store_trans_df.head()


# In[258]:


store_bar_sales.add_trace(px.scatter(store_trans_df, x='store_nbr', y='transactions').data[0])
store_bar_sales.show()
#transactions_df.set_index('date').groupby('store_nbr').resample('D').transactions.sum().reset_index()
#trans_df.head()


# In[ ]:


px.scatter(store_trans_df, x='', y='transactions').data[0])


# In[ ]:





# ### 6. Merged dataset

# Here, lets merge all the datasets on date and store number.

# In[162]:


merged_train_df.shape


# In[163]:


merged_train_df.head()


# In[164]:


merged_train_df.info()


# In[165]:


merged_train_df.columns


# In[166]:


#lets drop some of the categorical columns 

reduced_train_df = merged_train_df.drop(columns=['id', 'date', 'family', 'type_x',
       'locale', 'locale_name', 'description', 'city', 'state', 'type_y', 'cluster', 'Day','Month',
       'Year'])
reduced_train_df.head()


# In[167]:


reduced_train_df.corr().style.background_gradient(cmap='Oranges')


# r-values greater than 0.7 indicate a strong correlation between two attributes. From the graph above, there doesn't seem to be any strong correlations. However, the graph does confirm our findings in the data analysis: -
# 1. transactions and sales are positively correlated
# 2. promotions and sales are positively correlated
# 3. transactions and store numbers are positively correlated 

# In[168]:


merged_train_df['locale'].unique()


# In[169]:


merged_train_df['type_y'].unique()


# In[170]:


merged_train_df['type_x'].unique()


# In[203]:


#make a copy of df

merged_train_df2 = merged_train_df.copy()


# In[204]:


mapping_dict1 = {'National':1, 'Local':2, 'Regional':3}
merged_train_df2['locale'] = merged_train_df2['locale'].map(mapping_dict1)

mapping_dict2 = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
merged_train_df2['type_y'] = merged_train_df2['type_y'].map(mapping_dict2)

mapping_dict3 = {'Holiday':1, 'Work Day':2, 'Additional':3, 'Event':4, 'Transfer':5, 'Bridge':6}
merged_train_df2['type_x'] = merged_train_df2['type_x'].map(mapping_dict3)


# In[172]:


merged_train_df['family'].unique()


# In[205]:


fam_d={}
for index, key in enumerate(merged_train_df2['family'].unique(), start=1):
    fam_d[key] = index

print(fam_d)


# In[206]:


merged_train_df2['family'] = merged_train_df2['family'].map(fam_d)


# In[207]:


merged_train_df2.head()


# In[208]:


#lets expand on our reduced df to include some of the categorical data

reduced_train_df2 = merged_train_df2.drop(columns=['id', 'date', 
      'locale_name', 'description', 'city', 'state', 'cluster', 'Day','Month',
       'Year'])
reduced_train_df2.head()


# In[209]:


reduced_train_df2.corr().style.background_gradient(cmap='Greens')


# By expanding the dataset to include some categorical data, it does not seem to show any further correlations between data attributes. 

# In[ ]:


columns_to_show=['id', 'date', 'family', 'sales', 'onpromotion', 'type_x',
       'locale', 'locale_name', 'description', 'transferred', 'dcoilwtico',
       'city', 'state', 'type_y', 'cluster', 'transactions', 'Day', 'Month',
       'Year']


# In[243]:


stores = merged_train_df.groupby(['store_nbr'])[['sales', 'transactions']].sum().round(2).reset_index()
stores


# In[250]:


fig = px.bar(stores, x='store_nbr', title = "Sales and Transactions by Store", color= stores['sales'])
fig.show()


# ### 6a. Time Related Features

# In[219]:


#time-step feature

#create a time dummy by counting out the length of the series

sales_by_store.head()


# In[217]:


sales_by_store.index


# In[230]:


sales_by_store2 = sales_by_store.copy()
sales_by_store2.head()


# In[232]:


summary = sales_by_store2.groupby(['store_nbr','date'])['sales'].sum().round(2).reset_index()
summary.head()


# In[233]:


summary2 = summary.set_index('date')
summary2.head()


# In[235]:


summary2['Time'] = np.arange(len(summary2.index))
summary2.head()


# In[236]:


store1 = summary2.loc[summary2['store_nbr']==1]
store1

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

fig, ax = plt.subplots()
ax.plot('Time', 'sales', data=summary2, color='0.75')
ax = sns.regplot(x='Time', y='sales', data=summary2, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Product Sales');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#archived!!


# In[ ]:


#let's assign numerical values to some of the categorical data to carry out more in-depth analysis (i.e. family)

merged_train_df['family'].unique()


# In[ ]:


fam_d={}
for index, key in enumerate(merged_train_df['family'].unique(), start=1):
    fam_d[key] = index

print(fam_d)


# In[ ]:


merged_train_df['family'] = merged_train_df['family'].map(fam_d)
merged_train_df.head()


# In[ ]:


merged_train_df['family'].unique()


# In[ ]:


merged_train_df['type_x'].unique()


# In[ ]:


hol_d={}
for index, key in enumerate(merged_train_df['type_x'].unique(), start=1):
    hol_d[key] = index

print(hol_d)


# In[ ]:


merged_train_df['type_x'] = merged_train_df['type_x'].map(hol_d)
merged_train_df.head()


# In[ ]:


merged_train_df['locale'].unique()


# In[ ]:


loc_d={}
for index, key in enumerate(merged_train_df['locale'].unique(), start=1):
    loc_d[key] = index

print(loc_d)


# In[ ]:


merged_train_df['locale'] = merged_train_df['locale'].map(loc_d)
merged_train_df.head()


# In[ ]:


merged_train_df['locale_name'].unique()


# In[ ]:


loc_name_d={}
for index, key in enumerate(merged_train_df['locale_name'].unique(), start=1):
    loc_name_d[key] = index

print(loc_name_d)


# In[ ]:


merged_train_df['locale_name'] = merged_train_df['locale_name'].map(loc_name_d)
merged_train_df.head()


# In[ ]:


merged_train_df['description'].unique()


# In[ ]:





# In[ ]:


#Let's take a closer look at the spread of the sales data 
plt.figure(figsize = (12,8))

sns.boxplot(data=train_data, y ='sales', showmeans=True)

mean_sales=train_data['sales'].mean()
median_sales=train_data['sales'].median()
plt.axhline(y=mean_sales, color='r', linestyle='-')
plt.axhline(y=median_sales, color='g', linestyle='-')

plt.title('Sales Data')

plt.show()


# In[ ]:


train_data.groupby('family')['sales'].sum()


# In[ ]:


train_data['family'].nunique()


# In[ ]:


train_data['sales'].value_counts()

