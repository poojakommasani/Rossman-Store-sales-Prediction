#!/usr/bin/env python
# coding: utf-8

# # Import libraries and data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


train = pd.read_csv("train.csv", parse_dates = True, dtype={'StateHoliday': str})
store = pd.read_csv("store.csv")


# # Describe data

# In[3]:


train.head()


# In[4]:


train.describe().apply(lambda s: s.apply('{:.2f}'.format))


# In[5]:


corr = train.corr().round(2)
mask = np.triu(np.ones_like(corr, dtype=np.bool))
sns.heatmap(corr, annot = True, mask = mask)

plt.show()


# # Customers

# In[6]:


plt.hist(train['Customers'], 100)
plt.axvline(train['Customers'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()

print(f"Mean: {train['Customers'].mean()}")


# In[7]:


plt.scatter(train['Sales'], train['Customers'], s=1)
plt.xlabel("Sales")
plt.ylabel("Customers")
plt.show()


# # Open vs Closed days

# In[8]:


open_days = train[train['Open'] == 1]
close_days = train[train['Open'] == 0]

print(f'Open days: {open_days.shape[0]}')
print(f'Closed days: {close_days.shape[0]}')


# In[41]:


open_days_no_sales = open_days[open_days['Sales'] == 0]
close_days_with_sales = close_days[close_days['Sales'] != 0]

print(f'Open days with no sales: {open_days_no_sales.shape[0]}')
print(f'Close days with sales: {close_days_with_sales.shape[0]}')

# open_days[open_days['Sales'] == 0]['Store'].value_counts()


# # Distribution of Sales

# In[10]:


plt.hist(open_days['Sales'], 100)
plt.axvline(open_days['Sales'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()

print(f"Mean: {open_days['Sales'].mean()}")


# # Avg Customers vs Store sales

# In[42]:


avg_customers_per_store = open_days.groupby('Store')['Customers'].mean()
total_sales_per_store = open_days.groupby('Store')['Sales'].sum()

plt.scatter(avg_customers_per_store, total_sales_per_store, s=1)
plt.xlabel("Avg customer per store")
plt.ylabel("Total sale per store")
plt.show()


# # Sales per month

# In[12]:


open_days['Month'] = pd.DatetimeIndex(open_days['Date']).month
sales_per_month = open_days.groupby('Month')['Sales'].mean()

plt.plot(sales_per_month)
plt.title('Sales per month')
plt.show()


# # Days of the week

# In[13]:


days_of_week = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
per_day = [open_days[open_days['DayOfWeek'] == day] for day in days_of_week]


# In[14]:


num_open_days = [day.shape[0] for day in per_day]

plt.title('Open days')
plt.bar(days_of_week.values(), num_open_days)
plt.show()


# In[15]:


sales_per_day = [day['Sales'].mean() for day in per_day]

plt.title('Avg sales per open day')
plt.bar(days_of_week.values(), sales_per_day)
plt.show()


# # Promos

# In[16]:


open_with_promo = open_days[open_days['Promo'] == 1]['Sales']
open_without_promo = open_days[open_days['Promo'] == 0]['Sales']


# In[17]:


plt.title('Number of days with promo vs no promo')
plt.bar(['With promo', 'Without promo'], [open_with_promo.shape[0], open_without_promo.shape[0]])
plt.show()

print(f'Days with promo: {open_with_promo.shape[0]}')
print(f'Days without promo: {open_without_promo.shape[0]}')


# In[18]:


plt.title('Avg sales with vs without promo')
plt.bar(['With promo', 'Without promo'], [open_with_promo.mean(), open_without_promo.mean()])
plt.show()

print(f'Avg sales with promo: {open_with_promo.mean()}')
print(f'Avg sales without promo: {open_without_promo.mean()}')


# # State Holidays

# In[19]:


types_of_holidays = {'a': 'public', 'b': 'easter', 'c': 'christmas', '0': 'none'}
only_holidays = {k: types_of_holidays[k] for k in types_of_holidays if k != '0'}


# In[20]:


num_per_holiday = [open_days[open_days['StateHoliday'] == holiday].shape[0] for holiday in only_holidays]

plt.title('Number of days per holiday type')
plt.bar(only_holidays.values(), num_per_holiday)
plt.show()

print(f'Public holidays: {num_per_holiday[0]}')
print(f'Easter holidays: {num_per_holiday[1]}')
print(f'Christmas holidays: {num_per_holiday[2]}')


# In[21]:


sales_per_holiday = [open_days[open_days['StateHoliday'] == holiday]['Sales'].mean() for holiday in types_of_holidays]

plt.title('Avg sales per holiday type')
plt.bar(types_of_holidays.values(), sales_per_holiday)
plt.show()


# # School Holidays

# In[22]:


school = open_days[open_days['SchoolHoliday'] == 1]['Sales']
no_school = open_days[open_days['SchoolHoliday'] == 0]['Sales']
school_and_state = open_days[(open_days['SchoolHoliday'] == 1) & (open_days['StateHoliday'] != '0')]

print(f'School holiday: {school.shape[0]}')
print(f'School holiday and state holiday: {school_and_state.shape[0]}')


# In[23]:


plt.title('Avg sales with vs without school holidays')
plt.bar(['School Holiday', 'No School Holiday'], [school.mean(), no_school.mean()])
plt.show()

print(f'Avg sales with promo: {school.mean()}')
print(f'Avg sales without promo: {no_school.mean()}')


# # Store supplemental data

# In[24]:


store.describe()


# In[25]:


store.isnull().sum()


# In[26]:


combined = train.merge(store)
combined.head()


# In[27]:


combined_open = open_days.merge(store)


# # Store Type

# In[28]:


types_of_stores = ['a','b','c','d']
sales_per_store_type = [combined_open[combined_open['StoreType'] == store_type]['Sales'].mean() for store_type in types_of_stores]

plt.title('Avg sales per store type')
plt.bar(types_of_stores, sales_per_store_type)
plt.show()


# # Assortment

# In[29]:


assortments = ['a','b','c']
sales_per_assortment = [combined_open[combined_open['Assortment'] == assortment]['Sales'].mean() for assortment in assortments]

plt.title('Avg sales per store assortment')
plt.bar(assortments, sales_per_assortment)
plt.show()


# # Competition Distance

# In[30]:


plt.hist(combined['CompetitionDistance'], 100)
plt.axvline(combined['CompetitionDistance'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()

print(f"Mean: {combined['CompetitionDistance'].mean()}")


# # Competition Open

# In[31]:


plt.hist(combined['CompetitionOpenSinceYear'], 100)
plt.axvline(combined['CompetitionOpenSinceYear'].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()

print(f"Mean: {combined['CompetitionDistance'].mean()}")


# # Promos

# In[32]:


open_with_promo2 = combined_open[combined_open['Promo2'] == 1]['Sales']
open_without_promo2 = combined_open[combined_open['Promo2'] == 0]['Sales']


# In[33]:


plt.title('Number of days with promo2 vs no promo2')
plt.bar(['With promo2', 'Without promo2'], [open_with_promo2.shape[0], open_without_promo2.shape[0]])
plt.show()

print(f'Days with promo: {open_with_promo2.shape[0]}')
print(f'Days without promo: {open_without_promo2.shape[0]}')


# In[34]:


plt.title('Avg sales with vs without promo2')
plt.bar(['With promo2', 'Without promo2'], [open_with_promo2.mean(), open_without_promo2.mean()])
plt.show()

print(f'Avg sales with promo2: {open_with_promo2.mean()}')
print(f'Avg sales without promo2: {open_without_promo2.mean()}')


# In[35]:


months = ['Jan', 'Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
promo2_per_month = store['PromoInterval'].str.get_dummies(sep=',').sum()
promo2_per_month = promo2_per_month.reindex(index = months)

plt.title('Promo2 renewals')
plt.bar(months, promo2_per_month)
plt.show()

