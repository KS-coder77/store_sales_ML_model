#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # Introduction 

# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRO31PUyfyM9_vfCLlaPGbQuaUXdP2IjlkaD4fEnhvBKvJ0ur8U54G4EMqDN4ffyFGEDU0&usqp=CAU)

# #### Define The Problem ...
# 
# Predict the next months sales figures for thousands of product families (e.g. Automotive, Babycare, Beauty, Books etc.) sold in various Corporacion Favorita stores located in Ecuador. The training data includes past sales data from 2013 to 2017. Supplementary information is provided in other files which can be used to help build the model.
# 
# #### Understand the Type of Problem ...
# 
# Seeing as we need to predict future sales (i.e. the next 15 days) using observed data, this would call for regression analysis. Regression analysis can help decipher what factors affect sales figures and help us build a model based on these factors.
# 

# ### Datasets
# 
# #### 1. train.csv
# The training data, comprising time series of features 
# * store_nbr: identifies the store at which the products are sold.
# * family: identifies the type of product sold.
# * onpromotion: gives the total number of items in a product family that were being promoted at a store at a given date.
# * sales: gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
# 
# #### 2. test.csv
# - The test data, having the same features as the training data. You will predict the target sales for the dates in this file.
# - The dates in the test data are for the 15 days after the last date in the training data.
# 
# #### 3. stores.csv
# - Store metadata, including city, state, type, and cluster.
# - cluster is a grouping of similar stores.
# 
# #### 4. oil.csv
# Daily oil price. Includes values during both the train and test data timeframes. (Ecuador is an oil-dependent country and it's economical health is highly vulnerable to shocks in oil prices.)
# 
# #### 5. holidays_events.csv
# Holidays and Events, with metadata
# 
# ##### Additional Notes
# 
# - Wages in the public sector are paid every two weeks on the 15th and on the last day of the month. Supermarket sales could be affected by this.
# - A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake.
# 
# #### NOTE: 
# Pay special attention to the transferred column. A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government. A transferred day is more like a normal day than a holiday. To find the day that it was actually celebrated, look for the corresponding row where type is Transfer. For example, the holiday Independencia de Guayaquil was transferred from 2012-10-09 to 2012-10-12, which means it was celebrated on 2012-10-12. Days that are type "Bridge" are extra days that are added to a holiday (e.g., to extend the break across a long weekend). These are frequently made up by the type "Work Day" which is a day not normally scheduled for work (e.g., Saturday) that is meant to payback the Bridge.
# 
# Additional holidays are days added to a regular calendar holiday, for example, as typically happens around Christmas (making Christmas Eve a holiday).

# ### Libraries

# In[3]:


#import libraries
import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

import plotly as py
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objs as go

sns.set_theme(style='darkgrid')


# # Clean Data

# In[4]:


#train_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/train.csv")
#test_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/test.csv")
#hol_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv")
#oil_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/oil.csv")
#stores_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/stores.csv")
#transactions_df = pd.read_csv("/kaggle/input/store-sales-time-series-forecasting/transactions.csv")
train_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\train.csv")
test_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\test.csv")
hol_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\holidays_events.csv")
oil_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\oil.csv")
stores_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\stores.csv")
transactions_df = pd.read_csv(r"C:\Users\Krupa\Downloads\store-sales-time-series-forecasting\transactions.csv")


# In[5]:


print("Train Data")
display(train_df.head())
print("Test Data")
display(test_df.head())
print("Holiday Events")
display(hol_df.head())
print("Oil Data")
display(oil_df.head())
print("Stores Data")
display(stores_df.head())
print("Transactions Data")
display(transactions_df.head())


# In[6]:


def check_data(df):
    """
    Generates a concise summary of DataFrame columns.
    """
    # Use list comprehension to iterate over each column
    summary = [
        [col, df[col].dtype, df[col].count(), df[col].nunique(), df[col].isnull().sum(), df.duplicated().sum()]
        for col in df.columns
    ]

    # Create a DataFrame from the list of lists
    df_check = pd.DataFrame(summary, columns=["column", "dtype", "instances", "unique", "sum_null", "duplicates"])

    return df_check


# In[7]:


print("Training Data Summary")
display(check_data(train_df))
print("Test Data Summary")
display(check_data(test_df))
print("Holidays Events Data Summary")
display(check_data(hol_df))
print("Oil Data Summary")
display(check_data(oil_df))
print("Stores Data Summary")
display(check_data(stores_df))
print("Transactions Data Summary")
display(check_data(transactions_df))


# From the summaries above, there appears to be some missing values in the oil prices dataset, but overall the datasets are in good shape.

# In[8]:


train_df.describe()


# In[9]:


train_df.hist(bins=50, figsize=(12,8))


# In[10]:


#let's change the date columns in each of the datasets to datetime 
train_df['date'] = pd.to_datetime(train_df['date'], format = "%Y-%m-%d")
test_df['date'] = pd.to_datetime(test_df['date'], format = "%Y-%m-%d")
hol_df['date'] = pd.to_datetime(hol_df['date'], format = "%Y-%m-%d")
oil_df['date'] = pd.to_datetime(oil_df['date'], format = "%Y-%m-%d")
transactions_df['date'] = pd.to_datetime(transactions_df['date'], format = "%Y-%m-%d")


# ### Fill missing values in Oil Dataset 

# In[11]:


oil_df.head()


# In[12]:


missing_oil_vals = oil_df.isna().sum()
missing_oil_vals


# In[13]:


sample_mean = np.mean(oil_df['dcoilwtico'].dropna())
sample_mean


# In[14]:


sample_med = np.median(oil_df['dcoilwtico'].dropna())
sample_med


# In[15]:


sample_mode = stats.mode(oil_df['dcoilwtico'].dropna())
sample_mode


# In[16]:


plt.figure(figsize=(12, 8))

sns.kdeplot(x='dcoilwtico', data = oil_df.dropna())

plt.title('Distribution of Oil Price', fontsize='15')

plt.axvline(x=sample_mean, color='yellow', linestyle='-')
plt.axvline(x=sample_med, color='r', linestyle='-.')

plt.xlabel('Values', fontsize='15')
plt.ylabel('Probability', fontsize='15')

plt.show()


# From the above kde plot, the oil price seems to be bimodal (i.e. two groups of oil prices). It is difficult to judge if an extreme price is due to local or gloabal factors. Also, the median is lower than the mean value, therefore suggesting a positive skewness.
# 
# Let's use interpolation to fill the missing oil price values.

# In[17]:


#let's fill the missing values in the oil price column using linear interpolation 
oil_df["dcoilwtico"] = np.where(oil_df["dcoilwtico"] == 0, np.nan, oil_df["dcoilwtico"])
oil_df["dcoilwtico_interpolated"] =oil_df.dcoilwtico.interpolate()


# In[18]:


oil_df.head()


# In[19]:


#for ease we will use the nearest figure to fill the first nan
oil_df['dcoilwtico_interpolated'].fillna(93.14, inplace=True)


# In[20]:


oil_df.head(25)


# In[21]:


oil_df.isna().sum()


# # Data Analysis 

# ### 1. Train data

# In[22]:


train_df.head()


# In[23]:


zero_sales_df = train_df.loc[train_df['sales']== 0.0]
zero_sales_df


# In[24]:


#Some stores have zero sales for certain product families (i.e. store number 1 has zero automotive sales)
#zero sales as a percent of overall sales 

total_sales = train_df['sales'].count()
zero_sales = (train_df['sales'] == 0.0).sum()

print("Percentage of zero sales overall: ", ((zero_sales/total_sales)*100).round(2),"%")


#  Approximately a third of the dataset has zero sales. This could be relevant information, so we will keep these rows.

# ## DO WE NEED THE ROW OF CODE BELOW - reduced_train_df??

# In[25]:


reduced_train_df = train_df.loc[train_df['sales']> 0.0]
reduced_train_df


# In[26]:


#let's split the date column into day/month/year in the training data
train_df['Day of Week'] = train_df['date'].dt.dayofweek
train_df['Day Name'] = train_df['date'].dt.day_name()
train_df['Month'] = pd.to_datetime(train_df['date'], format = '%m').dt.month
train_df['Year'] = pd.to_datetime(train_df['date'], format='%Y').dt.year


# In[27]:


train_df.head()


# In[28]:


train_data = train_df.copy()
train_data.set_index('date', inplace=True)


# In[29]:


# 19/06/24
# analyse training date - prod. family insights, sales and promos insights 

# begin by looking at monthly total sales figures, then drill down into weekly avg sales and daily total sales and daily avg sales for even more granular normalisation if necessary ..

# use total sales to understand overall performance 

# use avg sales to compare periods of different lengths and to normalise data for comparative analysis 


# In[30]:


#monthly total sales 
monthly_total_sales = train_data.resample('M').sales.sum().reset_index()
monthly_total_sales


# In[31]:


plt.figure(figsize=(12, 8))
ax=sns.lineplot(x='date', y='sales', data=monthly_total_sales)
#non-log labels
#ticks=[0, 2500000, 5000000, 7500000, 10000000, 12500000, 15000000, 17500000, 20000000, 22500000, 25000000, 27500000, 30000000, 32500000]
#ticks2=[0, 5, 10, 15, 20, 25, 30, 35]
#ax.set_yticks(ticks2)
#ax.set_yticklabels(ticks2)
ax.set(ylabel = 'Sales (ten million)')
plt.title('Line plot of Total Monthly Sales')
plt.show()


# In[32]:


# weekly avg sales 
weekly_avg_sales = train_data.resample('W').sales.mean().reset_index()
weekly_avg_sales


# In[33]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='date', y='sales', data=weekly_avg_sales)
plt.title("Line plot of Average Weekly Sales")
plt.show()


# In[34]:


#monthly average sales 
monthly_avg_sales = train_data.resample('M').sales.mean().reset_index()
monthly_avg_sales


# In[35]:


monthly_avg_sales['year'] = monthly_avg_sales['date'].dt.year
monthly_avg_sales['month'] = monthly_avg_sales['date'].dt.strftime('%B')


# In[36]:


monthly_avg_sales


# In[37]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='month', y='sales', hue = 'year', palette = 'deep', data=monthly_avg_sales)
plt.title("Monthly Average Sales by Year")
plt.show()


# In[38]:


# daily total sales 
daily_total_sales = train_data.resample('D').sales.sum().reset_index()
daily_total_sales


# In[39]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='date', y='sales', data=daily_total_sales)
plt.title('Line plot of Total Daily Sales')
plt.show()


# In[40]:


#daily average sales
daily_avg_sales = train_data.resample('D').sales.mean().reset_index()
daily_avg_sales


# In[41]:


daily_avg_sales['year'] = daily_avg_sales['date'].dt.year
daily_avg_sales['day_name'] = daily_avg_sales['date'].dt.strftime('%A')
daily_avg_sales['day'] = daily_avg_sales['date'].dt.day
daily_avg_sales['month'] = daily_avg_sales['date'].dt.strftime('%B')
daily_avg_sales


# In[42]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='day_name', y='sales', hue = 'year', palette = 'deep', data=daily_avg_sales)
plt.title('Line plot of Average Daily Sales')
plt.show()


# In[43]:


plt.figure(figsize=(12, 8))
sns.lineplot(x='day', y='sales', hue = 'year', palette = 'pastel', data=daily_avg_sales)
plt.title('Line plot of Average Daily Sales')
plt.show()


# To summarise, 
# 
# - monthly total sales - 2013 and 2014 seem to be quite volatile years. The sales figures fluctuate a lot. Sales seem to grow steadily from mid-2015 onwards. 
# - weekly average sales - suggest that sales increase towards the end of the week 
# - daily total sales - the start of the year seems to be slow, and then stadily picks up pace
# - daily average sales - sales have ...
# 
# 

# In[44]:


# Wages in the public sector are paid every two weeks on the 15 th and on the last day of the month. Supermarket sales could be affected by this.


# In[45]:


hol_df.shape


# In[46]:


# further EDA 25/06/24

# join train data and hols data - consider total sales on transfered hol days and bridge dates 
# ditto promos 

train_hols_df = pd.merge(train_df, hol_df, how='inner', on ='date')
train_hols_df


# In[47]:


#look at sales and promos for each prod. family

prod_family = train_hols_df.groupby('family').agg({'onpromotion': 'sum', 'sales': 'mean'}).reset_index()
prod_family 


# In[48]:


plt.figure(figsize=(12, 8))
sns.regplot(x='onpromotion', y='sales', data=prod_family)
plt.xticks(rotation=45)
plt.title('Relationship between Sales and Promotions')
plt.show()


# In[49]:


sales_fam = train_hols_df.loc[train_hols_df['type'] == 'Bridge'].groupby('family').sales.mean().sort_values(ascending=False).reset_index()
sales_fam


# In[50]:


sales_fam['family'].unique()


# In[51]:


plt.figure(figsize=(12, 8))
sns.barplot(x='family', y='sales', data=sales_fam.head(10))
plt.xticks(rotation=45)
plt.title('Top Product Family Sales on Bridge dates')
plt.show()


# In[52]:


#top product families on bridge dates tend to be groceries and beverages, followed by cleaning products and fresh produce, dairy and baked goods. 


# In[53]:


promo_fam = train_hols_df.loc[train_hols_df['type'] == 'Bridge'].groupby('family').onpromotion.sum().sort_values(ascending=False).reset_index()
promo_fam


# In[54]:


plt.figure(figsize=(12, 8))
sns.barplot(x='family', y='onpromotion', data=promo_fam.head(10))
plt.xticks(rotation=45)
plt.title('Top Product Families on Promotion on Bridge dates')
plt.show()


# In[55]:


plt.figure(figsize=(12, 8))
sns.barplot(x='family', y='onpromotion', data=prod_family.sort_values(by='onpromotion', ascending=False).head(10))
plt.xticks(rotation=45)
plt.title('Top Product Families on Promotion')
plt.show()


# In[56]:


plt.figure(figsize=(12, 8))
sns.barplot(x='family', y='sales', data=prod_family.sort_values(by='sales', ascending=False).head(10))
plt.xticks(rotation=45)
plt.title('Top Product Families by Sales Figures')
plt.show()


# In[57]:


train_hols_df.loc[train_hols_df['type'] == 'Bridge']


# In[58]:


train_hols_df2 = train_hols_df.copy()


# In[59]:


#rltnshp btwn sales and promos 

plt.figure(figsize=(12, 8))
sns.scatterplot(x='onpromotion', y='sales', data=train_hols_df, hue='type')
plt.xticks(rotation=45)
plt.title('Relationship between Sales and Promotions on Bridge dates')
plt.show()


# In[60]:


# analysis of sales and promos figures for each product family 

import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")


# Draw a categorical scatterplot to show each observation
ax = sns.swarmplot(data=df, x="body_mass_g", y="sex", hue="species")
ax.set(ylabel="")


# In[61]:


#mega join between train data, stores data and transactions data

combo_train_stores_df = pd.merge(train_hols_df, stores_df, how = 'inner', on ='store_nbr')
combo_train_stores_df


# In[62]:


combined_df = pd.merge(combo_train_stores_df, transactions_df, how='inner', on=['date', 'store_nbr'])
combined_df


# In[63]:


# sales, promos and trans by store / by date / by locale_name / by hol_type 

#1. store analysis 
store_sales = combined_df.groupby('store_nbr').sales.mean().reset_index()
store_sales


# In[64]:


plt.figure(figsize=(12, 8))
sns.barplot(x='store_nbr', y='sales', data=store_sales)
plt.xticks(rotation=45)
plt.title('Average Sales by Store')
plt.show()


# In[65]:


store_promos = combined_df.groupby('store_nbr').onpromotion.sum().reset_index()
store_promos


# In[66]:


plt.figure(figsize=(12, 8))
sns.barplot(x='store_nbr', y='onpromotion', data=store_promos)
plt.xticks(rotation=45)
plt.title('Total Promotions by Store')
plt.show()


# In[67]:


store_trans = combined_df.groupby('store_nbr').transactions.sum().reset_index()
store_trans


# In[68]:


plt.figure(figsize=(12, 8))
sns.barplot(x='store_nbr', y='transactions', data=store_trans)
plt.xticks(rotation=45)
plt.title('Total Transactions by Store')
plt.show()


# In[69]:


#rltnshp between average monthly sales and promotions 

#resample sales data on a monthly frequency 
monthly_avg_sales=train_data.resample('M').sales.mean().reset_index()


# In[70]:


monthly_avg_sales.head()


# In[71]:


px.scatter(monthly_avg_sales, x = "date", y = "sales", trendline = "ols", trendline_color_override = "red", title = 'Monthly Average Sales, 2013 - 2017')


# There is an upward trend in sales figures since 2013, with December out-performing other months. 

# In[72]:


#resample promo data on a monthly frequency 
monthly_avg_promos=train_data.resample('M').onpromotion.mean().reset_index()
monthly_avg_promos.head()


# In[73]:


merged = pd.merge(monthly_avg_sales, monthly_avg_promos, on='date', how='left')
merged.head()


# In[74]:


px.scatter(merged, x = "onpromotion", y = "sales", trendline = "ols", trendline_color_override = "red", title = 'Relationship between sales and promotions')


# There seems to be a positive correlation between sales and promotional items.

# In[75]:


#let's review avg. sales figures across months and days of the week
sales_df = train_df.pivot_table(index="Day Name", columns="Month", values='sales', aggfunc='mean')
row_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sales_df=sales_df.reindex(row_order)


# In[76]:


plt.figure(figsize=(12,6))
sns.heatmap(sales_df, cmap='Blues', annot=True, fmt='.1f')
plt.title('Average sales by Month and Day of the Week')
plt.show()


# From the heatmap above, it is clear that sales are above average (359.02) on the weekends. Also, sales are higher in the lead up to Christmas. (i.e. October to December).

# ### What is the difference between sales and transactions ?
# 
# - Sales - the total sales for a product family at a particular store on a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips)
# 
# 
# - Transactions - the total transactions at a particular store on a given date.

# In[77]:


#daily sales by store_nbr
avg_daily_sales_by_store = train_df.set_index('date').groupby('store_nbr').resample('D').sales.mean().reset_index()
avg_daily_sales_by_store.head()


# In[78]:


#daily sales by store_nbr
sales_by_store = train_df.set_index('date').groupby('store_nbr').resample('D').sales.sum().reset_index()
px.line(sales_by_store, x='date', y='sales', color='store_nbr', title='Daily Total Sales by Store Number')


# The graph above illustrates when peak sales occurred between 2013 and 2017. Namely, Store Number 2 had an amazing day of sales on May 2nd 2016.

# In[79]:


#daily average sales by product family
sales_by_family = train_df.set_index('date').groupby('family').resample('D').sales.sum().reset_index()
px.line(sales_by_family, x='date', y='sales', color='family', title='Daily Total Sales by Product Family')


# In[80]:


sales_by_store.sort_values('sales', ascending=False)


# From the table and graphs above, we can deduce that 2016 was a bumper year for sales, whereas 2013 was a slower year with sales figures rarely passing the 40k mark. 

# In[81]:


#let's visualise the top performing stores by average sales figures
avg_store_sales_df = train_df.groupby('store_nbr').sales.mean().sort_values(ascending=False).reset_index()
avg_store_sales_df.head()


# In[82]:


avg_store_sales = px.bar(avg_store_sales_df, y='sales', x='store_nbr',  title='Average Sales by Store')
avg_store_sales


# In[83]:


# top 5 performing stores
top_stores = avg_store_sales_df.sort_values(by='sales', ascending=False).head(5)
top_stores


# In[84]:


# bottom 5 performing stores
bottom_stores = avg_store_sales_df.sort_values(by='sales', ascending=False).tail(5)
bottom_stores


# ### IS TOTAL SALES DF NEEDED???

# In[85]:


total_sales = train_df.set_index('date').groupby('store_nbr').resample('D').sales.sum().reset_index()
total_sales.head()


# In[86]:


avg_sales_by_prodfamily = train_df.groupby('family').sales.mean().sort_values(ascending=False).reset_index()
avg_sales_by_prodfamily


# In[87]:


px.bar(avg_sales_by_prodfamily.head(10), y='family', x='sales', color='family', title='Top 10 Average Sales by Product Family')


# In[88]:


#sales_by_family = train_df.groupby('family').sales.mean().sort_values(ascending=False).head(10).reset_index()
#sales_by_family
px.bar(avg_sales_by_prodfamily.tail(10), y='family', x='sales', color='family', title='Bottom 10 Average Sales by Product Family')


# Overall, the best performing product families are:- 
#    1. Grocery I
#    2. Beverages
#    3. Produce
#    4. Cleaning
#    5. Dairy
#     
#     
# Whereas, the worst performing product families are:-
#    1. Books
#    2. Baby care
#    3. Home appliances
#    4. Hardware
#    5. Magazines 
#  

# ### 2. Holidays and Events dataset

# In[89]:


hol_df.head()


# In[90]:


fig = px.histogram(hol_df, x='type', title = "Holiday Type Transferred Histogram", color= hol_df['transferred'])
fig.show()


# The graph above summarises the following changes to holidays and events across 2012 - 2017: - 
# - 12 "holidays" were transferred (i.e. a holiday was moved to another date by the government)
# - 2 "bridge" days were added (i.e. extra days added to a holiday to extend the break across a long weekend)
# - 51 "additional" days were added to the calendar for typical holidays around Christmas for example
# - 5 "work days" a day not normally scheduled for work (e.g. Saturday) that is meant to payback the Bridge
# - 56 "events"
# 
# These changes are important to note as they could affect sales figures.

# In[91]:


#Let's take a closer look at the "transferred" holidays 
transferred_df = hol_df.loc[hol_df['transferred']== True]
transferred_df


# In[92]:


# let's see what dates these "transfered" holidays were changed to
changed_hols_df = hol_df.loc[hol_df['type']=='Transfer']
changed_hols_df


# In[93]:


transf_hol_dates=changed_hols_df['date']
transf_hol_dates.to_frame()


# In[94]:


bridge_df = hol_df.loc[hol_df['type']=='Bridge']
bridge_df


# In[95]:


bridge_dates = bridge_df['date']
bridge_dates.to_frame()


# In[97]:


#let's take a closer look at sales figures on these transferred holidays and bridge dates
transf_hol_dates


# In[100]:


avg_store_sales_df


# In[101]:


#let's take a closer look at sales figures on these transferred holidays and bridge dates
transf_hol_sales = avg_daily_sales_by_store[avg_daily_sales_by_store['date'].isin(transf_hol_dates)]
transf_hol_sales


# In[102]:


test = transf_hol_sales.groupby('date')['sales'].sum().round(2).to_frame().reset_index()
test


# In[103]:


total_sales = sales_by_store.groupby('date')['sales'].sum().round(2).to_frame().reset_index()
total_sales


# In[104]:


avg_daily_sales = np.mean(total_sales['sales'])
avg_daily_sales


# In[105]:


ax = sns.barplot(x='date', y='sales', data=test)
#non-log labels
ticks=[0, 250000, 500000, 750000, 1000000, 1250000]
ax.set_yticks(ticks)
ax.set_yticklabels(ticks)
plt.xticks(rotation=45)
plt.axhline(avg_daily_sales, c='k', ls='-', lw=2.5)
plt.title('Total Sales Figures on Transferred Holiday Dates')
plt.show()


# It is evident that the sales figures are typically a lot higher than the average on the dates where holidays have been "transferred".

# In[106]:


#let's see the split by year group
hol_df['Day'] = hol_df['date'].dt.day_name()
hol_df['Month'] = pd.to_datetime(hol_df['date'], format = '%m').dt.month_name()
hol_df['Year'] = pd.to_datetime(hol_df['date'], format='%Y').dt.year


# In[107]:


fig = px.histogram(hol_df, x='Year', title = "Holiday Type by Year", color= hol_df['type'], barmode='group')
fig.show()


# In[108]:


fig = px.histogram(hol_df, x='Month', title = "Holiday Type by Month", color= hol_df['type'], barmode='group')
fig.update_xaxes(categoryorder='array', categoryarray= ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
fig.show()


# December has by far the highest number of additional holidays with a small number also appearing in May and July. December also has the highest number of bridge days, followed by January and November. October has the highest number of transfer days and May has the highest number of events throughout the year. Also, November has the most general holidays.
# 

# ### 3. Oil Price dataset

# In[109]:


oil_df.head()


# The oil dataset provides daily oil prices from 1st Jan 2013 to 31st August 2017.

# In[110]:


oil = oil_df.set_index('date').dcoilwtico.resample('D').sum().reset_index()
oil.head()


# In[111]:


oil["dcoilwtico"] = np.where(oil["dcoilwtico"] == 0, np.nan, oil["dcoilwtico"])
oil["dcoilwtico_interpolated"] =oil.dcoilwtico.interpolate()


# In[112]:


oil.head(25)


# In[113]:


oil_int_df = oil.melt(id_vars=['date']+list(oil.keys()[5:]), var_name='Legend')
px.line(oil_int_df.sort_values(["Legend", "date"], ascending = [False, True]), x='date', y='value', color='Legend',title = "Daily Oil Price" )


# From the graph above, oil prices hit a peak in September 2013, and steadily fell from September 2014 to Jan 2015. The lowest period was Jan 2016. The earthquake in April 2016 did not have seem to have a big impact on oil prices.

# In[114]:


#let's replace the NAN values with the interpolated values
values = oil['dcoilwtico'].values
values


# In[115]:


indices_nan = np.isnan(values)
indices_nan


# In[116]:


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


# In[117]:


oil = oil.drop(columns = 'dcoilwtico_interpolated')
oil.head(25)


# Let's see if there is any correlation between oil prices and sales/transactions. History suggests that when oil prices are high, the economy is struggling and product prices are expected to be high and sales therefore low.

# In[118]:


#rltnshp between oil price and sales
oil_data = oil.copy()
oil_data.set_index('date', inplace=True)

monthly_avg_oil_price=oil_data.resample('M').dcoilwtico.mean().reset_index()

monthly_avg_oil_price.head()


# In[119]:


merged = pd.merge(monthly_avg_sales, monthly_avg_oil_price, on='date', how='left')
merged.head()


# In[120]:


px.scatter(merged, x = "dcoilwtico", y = "sales", trendline = "ols", trendline_color_override = "red", title = 'Relationship between sales and oil price')


# As history has shown us, there is a negative correlation between oil prices and sales figures.

# ### 4. Store dataset

# In[121]:


stores_df.head()


# In[122]:


fig = px.histogram(stores_df, x='city', title = "Store Type by City Histogram", color='type')
fig.show()


# The majority of stores are type D, followed by type C. With the majority of them located in the capital city, Quito. 

# In[123]:


fig = px.histogram(stores_df, x='state', title = "Store Type by State Histogram", color= 'type')
fig.show()


# ### 5. Transactions dataset

# In[124]:


transactions_df.head()


# In[125]:


trans_data = transactions_df.copy()
trans_data.set_index('date', inplace=True)
monthly_avg_trans = trans_data.resample('M').transactions.mean().reset_index()
monthly_avg_trans.head()


# In[126]:


monthly_avg_trans['year']= monthly_avg_trans.date.dt.year
monthly_avg_trans.head()


# In[127]:


px.line(monthly_avg_trans, x='date', y='transactions', color='year', title='Average Monthly Transactions by Year')


# December is a peak month for store transactions. It is also evident that there is an element of seasonality across the years.

# In[128]:


trans_data1 = transactions_df.copy()
trans_data1['year'] = trans_data1.date.dt.year
trans_data1['dayofweek'] = trans_data1.date.dt.dayofweek+1
trans_data1=trans_data1.groupby(['year', 'dayofweek']).transactions.mean().reset_index()
trans_data1.head()
#monthly_avg_trans.head()


# In[129]:


px.line(trans_data1, x='dayofweek', y='transactions', color='year', title='Average Monthly Transactions by Day of the Week')


# Thursday is typically not a busy day for transactions, whereas Friday and Saturday are high performers.

# In[130]:


#daily transactions by store_nbr
trans_df = transactions_df.set_index('date').groupby('store_nbr').resample('D').transactions.sum().reset_index()
trans_df.head()


# In[131]:


merged=pd.merge(train_df.groupby(['date', 'store_nbr']).sales.sum().reset_index(), trans_df, how='left')
merged.head()


# In[132]:


px.scatter(merged, y = "sales", x ="transactions", trendline = "ols", trendline_color_override = "red", title = 'Relationship between transactions and sales')


# There is a positive correlation between sales and transactions.

# In[133]:


#transactions_df['date'] = pd.to_datetime(transactions_df['date'])
transactions_df['Day'] = transactions_df['date'].dt.day_name()
transactions_df['Month'] = pd.to_datetime(transactions_df['date'], format = '%m').dt.month
transactions_df['Year'] = pd.to_datetime(transactions_df['date'], format='%Y').dt.year


# In[134]:


transactions_df.head()


# In[135]:


#transactions by year
transactions_df.groupby('Year')['transactions'].sum().to_frame()


# In[136]:


avg_trans = transactions_df.pivot_table(index="Day", columns="Month", values='transactions', aggfunc='mean')
row_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_trans = avg_trans.reindex(row_order)
avg_trans


# In[137]:


plt.figure(figsize=(12,6))
sns.heatmap(avg_trans, cmap='Reds', annot=True, fmt='.1f')
plt.title('Average Transactions by Month and Day of the Week')
plt.show()


# Like sales figures, transactions tend to be higher on the weekends and in the months of May and December. There seems to be a linear relationship between sales and transactions. 

# In[138]:


px.line(trans_df, x='date', y='transactions', color='store_nbr',title = "Daily Total Transactions by Store Number" )


# ### 6. Merged Datasets Analysis

# In[139]:


merged_stores_df = pd.merge(stores_df, train_df, on='store_nbr', how='left')
merged_stores_df.head()


# In[140]:


stores_trans_df = pd.merge(merged_stores_df, transactions_df, on=['date', 'store_nbr'], how='left')
stores_trans_df = stores_trans_df.drop(columns=['Day', 'Month_y', 'Year_y'])
stores_trans_df.head()


# In[141]:


stores_trans_df = stores_trans_df.rename(columns={'Month_x':'Month', 'Year_x':'Year'})


# In[142]:


stores_trans_df['Day'] = stores_trans_df['date'].dt.day_name()
stores_trans_df['Month'] = pd.to_datetime(stores_trans_df['date'], format = '%m').dt.month
stores_trans_df['Year'] = pd.to_datetime(stores_trans_df['date'], format='%Y').dt.year


# In[143]:


stores_trans_df = stores_trans_df.dropna()
stores_trans_df.head()


# In[144]:


cluster_annual_avg_sales = stores_trans_df.groupby(by=['cluster', 'Year'])['sales'].mean().to_frame().reset_index()
cluster_annual_avg_sales


# In[145]:


fig=px.line(cluster_annual_avg_sales, x='Year', y='sales', color='cluster', title='Average Annual Sales by Cluster, 2013-2017')
fig.show()


# Cluster 5 seems to out perform the others in terms of average sales numbers, followed by clusters 11, 14 and 8. 

# In[146]:


cluster_annual_avg_transactions = stores_trans_df.groupby(by=['cluster', 'Year'])['transactions'].mean().to_frame().reset_index()
cluster_annual_avg_transactions


# In[147]:


fig=px.line(cluster_annual_avg_transactions, x='Year', y='transactions', color='cluster', title='Average Annual Transactions by Cluster, 2013-2017')
fig.show()


# Clusters 5, 11, 14 and 8 are the top performing in average sales and transaction figures.

# Insights from EDA:- 
#     
#     1. Sales have been increasing steadily since 2013 
#     2. Typically, Friday to Sunday are the more popular days of the week to shop on 
#     3. There is evidence of seasonality as October to December are peak sales months
#     4. Product sales are positively correlated with promoted itmes 
#     5. It's reasonable to say that sales rise on the days following "pay day" (i.e. the 15th day and last day of the  month)
#     6. Sales tend to increase on and around "bridge" dates (i.e. when holidays have been transferred and additional days added to extend the break across a long weekend)
# 

# ## 7. Time related features 

# ### 7.1 Linear Regression with Time Series 
# ### 7.1.1 Fit a time-step to daily average sales 

# In[191]:


store_sales = train_df.copy()
store_sales = store_sales.drop(columns=['Day of Week', 'Day Name', 'Month', 'Year'])
store_sales


# In[ ]:


#train_df1 = train_df.groupby(['date']).sales.mean().round(2).reset_index()

#create a time dummy feature (i.e. counts off time steps in the series)
#average_sales = train_df1.copy()
#time = np.arange(len(average_sales.index))

#average_sales['time'] = time

#average_sales.set_index('date', inplace=True)
#average_sales.head()


# In[192]:


store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)

store_sales


# In[193]:


average_sales = store_sales.groupby('date').mean()['sales']
average_sales.head()


# In[194]:


average_sales_df = average_sales.to_frame()

#create a time dummy feature (i.e. counts off time steps in the series)
time = np.arange(len(average_sales.index))
average_sales_df['time'] = time

average_sales_df


# In[195]:


from sklearn.linear_model import LinearRegression

#create training data
X = average_sales_df.loc[:, ['time']] #feature
y = average_sales_df.loc[:, 'sales' ] #target

#train and fit the model 
model = LinearRegression()
model.fit(X,y)

#store the predicted values
y_pred = pd.Series(model.predict(X), index=X.index)


# In[196]:


plot_params = {"color":"grey", "style":".-"}
fig, ax = plt.subplots(figsize=(12,6))
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Average Daily Sales')


# ### 7.1.2 Fit a Lag feature to average daily sales

# In[197]:


#create a lag feature from the target 'sales' (shift series down by 1)
lag_1 = average_sales_df['sales'].shift(1)

average_sales_df['lag_1'] = lag_1

#create training data
X = average_sales_df.loc[:, ['lag_1']].dropna() #feature
#X.dropna(inplace=True) #drop missing values in feature set
y = average_sales_df.loc[:, 'sales' ] #target
y, X = y.align(X, join='inner') #drop corresponding values in target 'sales'

#train and fit the model 
model = LinearRegression()
model.fit(X,y)

#store the predicted values
y_pred = pd.Series(model.predict(X), index=X.index)


# In[198]:


fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Lag Plot of Average Daily Sales');


# The above lag plot suggests a positive relationship and therefore an observation can be predicted from previous observations.

# ### 7.2 Trend 
# - the trend component of a time series represents a persistent, long-term change in the mean of the series
# - it is the slowest moving part of a series 
# - we can use a moving average plot to determine what kind of trend exists in a time series
# 
# ### 7.2.1 Identify type of trend with a moving average plot
# 

# In[199]:


store_sales


# In[200]:


# a function to group data with certain frequency (useful when it's a large dataset with many rows)

def grouped(df, key, freq, col):
    df_grouped = df.groupby([pd.Grouper(key=key, freq=freq)]).agg(mean=(col, 'mean'))
    df_grouped = df_grouped.reset_index()
    
    return df_grouped


# In[201]:


#function to plot moving avg using rolling method 

def plot_moving_avg(df, key, freq, col, window, min_periods, ax, title):
    df_grouped = grouped(df, key, freq, col)
    moving_avg = df_grouped['mean'].rolling(window=window, center=True, min_periods=min_periods).mean()
    ax = df_grouped['mean'].plot(color='0.75', linestyle='dashdot', ax=ax)
    ax = moving_avg.plot(linewidth=3, color='b', ax=ax)
    ax.set_title(title, fontsize=18)


# In[202]:


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30,20))

plot_moving_avg(train_df, 'date', 'W', 'sales', 7, 4, axes[0], 'Weekly Sales Moving Average')
plot_moving_avg(transactions_df, 'date', 'W', 'transactions', 7, 4, axes[1], 'Weekly Transactions Moving Average')
plt.show()


# There is a steady upward trend in the moving average sales plot and signs of seasonality in the moving average transactions plot.

# In[204]:


plt.rc("figure", autolayout=True, figsize=(11,5))

moving_average = average_sales_df['sales'].rolling(
window=12,           #365-day window (useful for capturing long-term trends and seasonality)
center=True,          #puts the average at the centre of the window
min_periods=6,      #choose about half the window size
).mean()              #compute the mean     

ax = average_sales_df['sales'].plot(style='.', color='0.5')
moving_average.plot(ax=ax, linewidth=3, title='Product Sales - 365-Day Moving Average', legend=False)


# The smaller window size of 12, identifies a repeating up and down movement year after year (i.e. a short-term seasonal change). To visualise a trend, we take an average over a period longer than any seasonal period in the series. Let's take a look with a larger window size of 365 to smooth over the season within each year. 

# In[205]:


plt.rc("figure", autolayout=True, figsize=(11,5))

moving_average = average_sales_df['sales'].rolling(
window=365,           #365-day window (we have daily observations, so a larger window size is useful for capturing long-term trends and seasonality)
center=True,          #puts the average at the centre of the window
min_periods=183,      #choose about half the window size
).mean()              #compute the mean     

ax = average_sales_df['sales'].plot(style='.', color='0.5')
moving_average.plot(ax=ax, linewidth=3, title='Product Sales - 365-Day Moving Average', legend=True)


# Now we can identify a clear upward trend (i.e. consistently increasing values), let's attempt to model it using a time-step feature. 

# We will use a function from the statsmodels library called DeterministicProcess to engineer our time dummy. This function will help to avoid some tricky failure cases that can arise with time series and linear regression.  

# ### 7.2.2 Model Trend 

# In[206]:


average_sales_df.head()


# In[208]:



from statsmodels.tsa.deterministic import DeterministicProcess

y = average_sales_df['sales'].copy()  # the target

dp = DeterministicProcess(
    index=average_sales.index,  # dates from the training data
    constant=True,             # dummy feature for the bias (y_intercept)
    order=1,                    # the time dummy (1:linear trend, 2:quadratic, 3:cubic etc.)
    drop=True,                  # drop terms if necessary to avoid collinearity
)

# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

# "out of sample" creates forecasts the next 30 days following the in-sample date range
X_fore = dp.out_of_sample(steps=30)

#X.head()


# In[209]:


# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.

model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)


# In[211]:


ax = average_sales_df['sales'].plot(style=".", color="0.5", title="Product Sales - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")


# The trend discovered by our LinearRegression model is very similar to the moving average plot, which suggestes that a linear trend was the right decision in this case.
# 
# To make a forecast, we apply our model to "out of sample" features (i.e. times outside of the observation period of the training data)
# 

# ### 7.2.3 Forecast Trend

# In[212]:


y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
#y_fore.head()


# In[213]:


#let's plot a portion of the series to see the trend forecast for the next 30days

ax = average_sales_df['sales']["2017-02-01":].plot(title="Product Sales - Linear Trend Forecast", **plot_params)
ax = y_pred["2017-02-01":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()


# ### 7.3 Seasonality 
# 
# - occurs when there is a regular, periodic change in the mean of the series 
# - generally follows the clock and calendar (i.e. repetitions over a day/week/yr are common)
# - often driven by cycles of the natural world or by conventions of social behaviour surrounding dates/times
# 
# There are two kinds of features that model seasonality:- 
# 
# 1. Indicators - best for a season with few observations such as weekly or daily observations
# 2. Fourier features - best for a season with many observations such as annual season of daily observations 
# 
# 
# ### 7.3.1 Seasonal Indicators 
# 
# - the outcome is what you get if you treat a seasonal period as a categorical feature and apply one-hot encoding 
# - binary features that represent seasonal differences in the level of a time series 
# 

# In[214]:


#function to create seasonal plot 

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ax=ax,
        palette=palette,
        legend=False,
        errorbar=('ci', False)
        
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


# In[217]:


X=average_sales_df.copy().drop(columns=['time', 'lag_1']).loc['2017']
X


# In[218]:


# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
seasonal_plot(X, y="sales", period="week", freq="day", ax=ax0)
seasonal_plot(X, y="sales", period="year", freq="dayofyear", ax=ax1);


# ### 7.3.2 Periodogram
# 
# - identifies the strength of the frequencies in a time series 

# In[219]:


#function to create periodogram 

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# In[220]:


plot_periodogram(X.sales)


# From the periodogram above, it is clear that weekly sales contribute significantly to the overall sales figures. There is a strong weekly season, followed by bi-weekly and monthly seasons. This could be related to wages paid fortnightly. 

# ### 7.4  Create Seasonal Features
# 
# We can use Derterministic Process and CalendarFourier to create:- 
# 
# 1. indicators for weekly seasons 
# 2. Fourier features of order 4 for monthly seasons 
# 

# In[228]:


average_sales


# In[229]:


y = average_sales.copy().loc['2017']


# In[230]:


y


# In[231]:


from statsmodels.tsa.deterministic import CalendarFourier

fourier = CalendarFourier(freq='M', order =4)
dp = DeterministicProcess(
index=y.index, 
constant=True, 
order=1,
seasonal=True,
additional_terms=[fourier],
drop=True,)

X = dp.in_sample()


# In[233]:


model = LinearRegression().fit(X,y)
y_pred =  pd.Series(
model.predict(X),
index=X.index,
name='Fitted')

y_pred=pd.Series(model.predict(X), index=X.index)
ax=y.plot(**plot_params, alpha=0.5, title='Average Sales', ylabel='Sales')
ax=y_pred.plot(ax=ax, label='Seasonal')
ax.legend()


# Now let's deseasonalise the data ...

# In[234]:


y_deseason = y - y_pred

y_deseason


# In[237]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, sharey=True, figsize=(10,7))
ax1 = plot_periodogram(y, ax=ax1)
ax1.set_title('Sales Frequency Components')
ax2 = plot_periodogram(y_deseason, ax=ax2)
ax2.set_title('Deseasonalised')


# The sales frequency components periodogram mirrors the periodogram for seasonality. The deseasonalised periodogram shows some noise around weekly and semi-weekly. This could be related to National Holidays in Ecuador. Let's take a closer look at the holidays dataset. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 7.4  Time Series as Features
# 
# - serial dependence: where the observations in a series are correlated with previous observations at differnt lags (put simply, the value of a variable at a certain point in time is dependent on its previous values).
# - serial dependence helps to detect and understand the validity of statistical tests and the performance of forecasting models.
# - cycles are a common way to identify serial dependence. Unlike seasonal behaviour, cycles are not necessarily time dependent. Cycles are affected by what has happened in the recent past and are therefore more irregular than seasonality.
# - techniques such as Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are commonly used to visualise serial dependence in time series data.
# 
# Our goal is to forecast sales over the coming weeks. We can take two approaches: - 
# 
# 1. forecast sales using lag features 
# 2. forecast sales using lags of "on promotion" time series - (contains the number of items on a special promo on a particular day)
# 
# Not every product family has sales showing cyclic behaviour, and neither does the series of average sales. However, sales of school and office supplies show patterns of growth and decay. Let's model cycles in sales of the school and office supplies product family using lag features.

# In[109]:


#import libraries  

from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


# In[110]:


store_sales = train_df.copy()


# In[111]:


store_sales.head()


# In[112]:


store_sales.drop(columns=['id', 'Day of Week', 'Day Name', 'Month', 'Year'], inplace=True)


# In[113]:


store_sales.head()


# In[114]:


store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
store_sales.head()


# In[115]:


family_sales=(store_sales.groupby(['family', 'date']).mean().unstack('family').loc['2017', ['sales', 'onpromotion']])
family_sales.head()


# In[116]:


book_sales = family_sales.loc(axis=1)[:, 'BOOKS']
book_sales.head()


# In[117]:


supply_sales = family_sales.loc(axis=1)[:, 'SCHOOL AND OFFICE SUPPLIES']
y = supply_sales.loc[:, 'sales'].squeeze()

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True, 
    index=y.index,
    order=1,
    seasonal=True, 
    drop=True, 
    additional_terms=[fourier]
    )

X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

model = LinearRegression (fit_intercept=False)
model.fit(X_time, y)
y_deseason = y-model.predict(X_time)
y_deseason.name = 'sales deseasoned'

ax = y_deseason.plot()
ax.set_title('Sales of school and office supplies (deseasonalised)')


# ### 7.4.1 Plotting Cycles 
# 
# Let's isolate cyclic behaviour using a moving-average plot

# In[118]:


y_ma = y.rolling(window=7, center=True).mean()
ax = y_ma.plot()
ax.set_title('7-day Moving Average Plot')


# The moving-average plot resembles the plot of the deseasonalised series. Let's take a closer look at the deseasonalised series for serial dependence by using partial autocorrelation correlogram and lag plot.

# In[119]:


#create a function to make lag plots

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)                                # lag data 
    if standardize:                                  # standardized is mean-centered and scaled by standard deviation
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y 
    else:
        y_ = x
    corr = y_.corr(x_)                               # calculate correlation between x_ and y_
    if ax is None:                                   # plot graph  
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3)
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,                          # scatter plot with regression line
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,                   # option to plot a locally weighted scatterplot smoothing line 
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left")
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math 
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags/nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] *2, nrows *2+0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):        
        if k + 1 <= lags:                                                    #loop over each subplot and its corresponding index "k" 
            ax=lagplot(x, y, lag=k+1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag{k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    
    return fig


# In[120]:


plot_lags(y_deseason, lags=8, nrows=2)
plot_pacf(y_deseason, lags=8);


# ### 7.4.2 Examine serial dependence
# 
# The lag plots indicate that the relationship of school supply sales to its lags is mostly linear, while the partial autocorrelations suggest the dependence can The lag plots indicate that the relationship of school supply sales to its lags is mostly linear, while the partial autocorrelations suggest the dependence can be captured using lag 1. be captured using lag 1. 
# 
# A leading indicator provides "advance notice" of changes in the target. A useful leading indicator could be the "onpromotion" series. Since the company itself decides when to do a promotion, there's no worry about "lookahead leakage". For example, we could use Tuesday's onpromotion value to forecast sales on Monday. 
# 
# Let's take a closer look at leading and lagging values for onpromotion plotted against sales of school and office supplies. 
# 

# In[121]:


onpromotion = supply_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')


# In[122]:


#create a function to make lead plots

def leadplot(x, y=None, lead=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(-lead)                              # lead data 
    if standardize:                                  # standardized is mean-centered and scaled by standard deviation
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y 
    else:
        y_ = x
    corr = y_.corr(x_)                               # calculate correlation between x_ and y_
    if ax is None:                                   # plot graph  
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3)
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,                          # scatter plot with regression line
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,                   # option fo plot a locally weighted scatterplot smoothing line 
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left")
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lead {lead}", xlabel=x_.name, ylabel=y_.name)
    
    return ax

def plot_leads(x, y=None, lead=3, nrows=1, lagplot_kwargs={}, **kwargs):
    import math 
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lead/nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] *2, nrows *2+0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):        
        if k + 1 <= lead:                                                    # loop over each subplot and its corresponding index "k" 
            ax=leadplot(x, y, lead=k+1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lead{k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    
    return fig


# In[123]:


plot_leads(x=onpromotion.loc[onpromotion > 1], y=y_deseason.loc[onpromotion > 1], lead=3, nrows=1)


# The lag and lead plots indicate that the onpromotion values correlate with the supply sales. Therefore, both could be useful as time series features. 

# ### 7.4.2 Create time series features 
# 

# In[124]:


y_deseason


# In[125]:


onpromotion


# In[126]:


#function to make lags
def make_lags(ts, lags, lead_time=1, name='y'):
    return pd.concat(
    {
        f'y_lag_{i}' : ts.shift(i)
        for i in range(1, lags+1)
    },
    axis=1)


# In[127]:


#function to make leads
def make_leads(ts, leads, name='y'):
    return pd.concat(
    {
        f'y_lead_{i}' : ts.shift(-i)
        for i in range(1, leads+1)
    },
    axis=1)


# In[128]:


for i in range(1,2):
    print(i)


# In[129]:


#make features from y_deseason
X_lags = make_lags(y_deseason, lags=1)
X_lags


# In[130]:


onpromotion


# In[131]:


make_leads(onpromotion, leads=1)


# In[132]:


#make features from onpromotion
X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    make_leads(onpromotion, leads=1)],
    axis=1)

X_promo


# In[133]:


# features
X = pd.concat([X_time, X_lags, X_promo], axis=1).dropna()

#drop corresponding vals in target
y, X = y.align(X, join='inner')


# In[134]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, shuffle=False)

model=LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5

print(f'Training RMSLE: {rmsle_train: 5f}')
print(f'Validation RMSLE: {rmsle_valid: 5f}')

ax = y.plot(**plot_params, alpha=0.5, title='Average Sales', ylabel='items sold')
ax = y_fit.plot(ax=ax, label='Fitted', color='C0')
ax = y_pred.plot(ax=ax, label='Forecast', color='C3')
ax.legend()


# The test data has a RMSLE of 0.35 which indicates good performance by the model. 

# ## 8 Hybrid Models 
# 
# The most common strategy for combining hybrid models is a simple (usually linear) learning algorith followed by a complex, non-linear learner like GBDTS or a deep neural net. The simpler model is typically designed as a "helper" for the powerful complex algorithm that follows. 
# 
# Generally a regression algorithm can make predictions in two ways: - 
# 
# 1. transforming the features (i.e. linear regression or neural nets)
# 2. transforming the target (i.e. decision trees or nearest neighbour) 
# 
# The predictions of target transformers will always be bound within the range of the training set, whereas the predictions of feature transformers will generally go beyond the training set. 
# 
# For these reasons, we will use linear regression to extrapolate the trend, transform the target to remove the trend and then apply XGBoost to the de-trended residuals.
# 

# In[135]:


store_sales


# In[136]:


family_sales


# ### 8.1 Define Boosted Hybrid Class 

# In[137]:


# create a Boosted Hybrid for store sales datset 

class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None # store col names from fit method
        
        


# ### 8.2 Add fit method to Boosted Hybrid Class 

# In[138]:


# add fit method to Boosted Hybrid class

def fit(self, X_1, X_2, y):
    # fit self.model_1
    self.model_1.fit(X_1, y)
    
    y_fit = pd.DataFrame(
    # make predictions with self.model_1
    self.model_1.predict(X_1),
    index = X_1.index, columns = y.columns,
    )
    
    # calc. residuals
    y_resid = y - self.model_1.predict(X_1)
    y_resid = y_resid.stack().squeeze()
    
    # fit self.model_2 on residuals
    self.model_2.fit(X_2, y_resid)
    
    # save col. names for predict method 
    self.y_columns = y.columns
    
    self.y_fit = y_fit
    self.y_resid = y_resid
    
# add method to class 
BoostedHybrid.fit = fit   
    


# ### 8.3 Add fit method to Boosted Hybrid Class 

# In[139]:


# add predict method to Boosted Hybrid class

def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
    self.model_1.predict(X_1),
    index = X_1.index,
    columns = self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()  # wide to long df
    
    # add self.model_2 predictions to y_pred
    y_pred += self.model_2.predict(X_2)
    
    return y_pred.unstack() # long to wide 

# add method to class
BoostedHybrid.predict = predict 


# ### 8.4 Prepare data for training 

# In[140]:


# set up store sales data for training 
from sklearn.preprocessing import LabelEncoder
# target series 
y = family_sales.loc[:, 'sales']

# X_1 features for linear regression model 
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

# X_2 features for XGBoost model
X_2 = family_sales.drop('sales', axis=1).stack() # onpromotion feature 

# Label encoding for 'family'
le = LabelEncoder()
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])

# Label encoding for seasonality 
X_2['day'] = X_2.index.day # values are day of the month 


# ### 8.5 Train Boosted Hybrid Model

# In[141]:


# Create Linear Regression and XGBRegressor hybrid with Boosted Hybrid Class
from xgboost import XGBRegressor

model = BoostedHybrid(
    model_1 = LinearRegression(),
    model_2 = XGBRegressor()
)

# call fit method 
model.fit(X_1, X_2, y)

# call predict method
y_pred = model.predict(X_1, X_2)
y_pred = y_pred.clip(0.0)


# In[142]:


y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train, X1_valid = X_1[: "2017-07-01"], X_1["2017-07-02" :]
X2_train, X2_valid = X_2.loc[:"2017-07-01"], X_2.loc["2017-07-02":]


model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), **plot_params, alpha=0.5,
)
_ = y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)


# # Data Modelling

# Now, let's select the features for our model. Based on our analysis so far, sales are affected by: - 
# 
# - day of the week (weekends show stronger sales figures)
# - month in the year (October to December show stronger sales figures)
# - oil price (higher sales figures when oil price low (and vice versa)
# - promotions (higher sales figures when more promotions)
# 
# Also, it is fair to say previous sales figures provide a good indicator for future sales.

# ### Fitting the Model
# 
# As this is a regresison problem, let's consider using the Linear Regression ML model.

# In[143]:


#import libraries 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings("ignore")


# In[144]:


oil_data1 = oil_data.copy()
oil_data1 = oil_data1.reset_index()
oil_data1


# In[145]:


#merge interpolated oil dataset with training data
train_df = pd.merge(train_df, oil_data1, how='inner', on='date')
train_df   


# In[146]:


train_df['weekend'] = train_df['Day of Week'].isin([5, 6]).astype(int)
train_df.drop(columns=['date', 'Day Name', 'store_nbr', 'Year'], axis=1, inplace=True)


# In[147]:


train_df.head()


# ### Prepare test data 

# In[148]:


test_df['Day of Week'] = test_df['date'].dt.dayofweek
test_df['Day Name'] = test_df['date'].dt.day_name()
test_df['Month'] = pd.to_datetime(test_df['date'], format = '%m').dt.month
test_df['Year'] = pd.to_datetime(test_df['date'], format='%Y').dt.year


# In[149]:


test_df = pd.merge(test_df, oil_data1, how='inner', on='date')
test_df


# In[150]:


test_df['weekend'] = test_df['Day of Week'].isin([5, 6]).astype(int)
test_df.drop(columns=['date', 'Day Name', 'store_nbr', 'Year'], axis=1, inplace=True)


# In[151]:


label=LabelEncoder()
train_df.family = label.fit_transform(train_df.family)
train_df.head()


# In[152]:


label=LabelEncoder()
test_df.family = label.fit_transform(test_df.family)
test_df.head()


# In[153]:


X = train_df.drop(['sales'], axis=1)
y = train_df['sales'] #target


# In[154]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### Linear Regression

# In[155]:


#initialize and fit linear regression model 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE: ', rmse)


# In[156]:


train_df['sales'].describe()


# Sale values range from 0 to 12,4717 with a mean of 357 and standard deviation of 1,101. A RMSE of 999 isn't great.

# In[157]:


print('score on test: ', end="")
print(str(model.score(X_test, y_test)))
print('score on train: ', end="")
print(str(model.score(X_train, y_train)))


# #### XGBoost

# In[158]:


xg_reg = xgb.XGBRegressor(n_estimators = 300, max_depth = 9)
xg_reg.fit(X_train, y_train)

y_pred = xg_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE: ', rmse)


# In[159]:


print('score on test: ', end="")
print(str(xg_reg.score(X_test, y_test)))
print('score on train: ', end="")
print(str(xg_reg.score(X_train, y_train)))


# XGBRegressor performs a lot better than linear regression, with RMSE of 633.47 compared with 999.15. However, just one test score is not reliable so we will also use k-fold cross-validation. 

# # Cross-validation
# 
# The concept of k-fold cross-validation is to split the test date multiple times (k-number splits or folds) into different training and test sets, and then to take the mean of the scores. There will be an overlap in the training sets, but not the test sets. 
# 
# 
# We will also use cross_val_score to evaluate the score. 

# ### Cross-Validation with linear regression 

# In[160]:


model = LinearRegression()

#implement cross_val_score with the models (number of folds is variable cv)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)

rmse_cv_lr = np.sqrt(-scores)
print('Reg. rmse (Linear Regression Model): ', np.round(rmse_cv_lr, 2))
print('RMSE mean (Linear Regression Model)): %0.2f' % (rmse_cv_lr.mean()))


# ### Cross-validation with XGBoost

# In[161]:


model = xgb.XGBRegressor() 

#implement cross_val_score with the models (number of folds is variable cv)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=10)

rmse_cv = np.sqrt(-scores)
print('Reg. rmse (XGBoost Model): ', np.round(rmse_cv, 2))
print('RMSE mean (XGBoost Model): %0.2f' % (rmse_cv.mean()))


# Now that we have evaluated the ML models, the RMSE value is significantly less with the XGBoost ML model, which suggests it will produce more reliable predictions.
# 

# # Submission

# In[162]:


y_pred_test = xg_reg.predict(test_df)


# In[163]:


data = test_df
data['Predicted Sales'] = y_pred_test


# In[164]:


data['Predicted Sales'].describe()


# In[165]:


data[data['Predicted Sales'] < 0]


# In[166]:


#set any negative values to zero 
data.loc[data['Predicted Sales'] < 0, 'Predicted Sales'] = 0


# In[167]:


data


# In[168]:


submission = pd.DataFrame({'id': test_df['id'], 'sales': data['Predicted Sales']})
submission


# In[169]:


submission.to_csv('submission.csv', index=False)

