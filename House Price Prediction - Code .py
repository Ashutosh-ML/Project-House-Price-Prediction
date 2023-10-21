#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
data= pd.read_csv('Bengaluru_House_Data.csv')


# In[2]:


data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# # Total Value count in every column

# In[5]:


data['area_type'].value_counts()


# In[6]:


data['availability'].value_counts()


# In[7]:


data['bath'].value_counts()


# In[8]:


data['location'].value_counts()


# In[9]:


data['size'].value_counts()


# In[10]:


data['society'].value_counts()


# In[11]:


data['balcony'].value_counts()


# In[12]:


data['price'].value_counts()


# # Checking null values in each column

# In[13]:


data.isnull().sum()


# In[14]:


#we will be droping the column society, balcony due to number of missing value and it can effects our model 
# we will be droping the column availability,area_type because they are not much relevent to our model
data.drop(columns=['area_type','availability','society','balcony'], inplace=True)
data


# In[15]:


#Check the 'describe' function to get a better idea of the numerical values in the data.
data.describe(percentiles=[0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])

#As we can see in the data, outliers are present between 90% and 100% of the data


# In[16]:


data.info()


# In[17]:


data['location'].value_counts()


# In[18]:


#To filling in missing values in the location column, so I will replace them with the location that has the maximum count and 
#store it in the location column

data['location'] = data['location'].fillna('Whitefield')


# In[19]:


#Filling Missing Values in the 'Size' Column: I will replace them with the size that has the maximum count and store it in the 'Size' column
    
data['size'].value_counts()


# In[20]:


data['size'] = data['size'].fillna('2 BHK')


# In[21]:


#To fill missing values in the 'bath' column, we will use the KNN imputer technique

from sklearn.impute import KNNImputer
knn_imputer = KNNImputer(n_neighbors=5)
data['bath'] = knn_imputer.fit_transform(data[['bath']])


# In[22]:


data.info()


# In[23]:


#Currently, the 'BHK' column contains a mix of data, including both numerical and string values. Therefore, I will separate 
#the BHK data based on their types, as it includes both integers and strings

data['bhk']=data['size'].str.split().str.get(0).astype(int)


# In[24]:


data['bhk']


# In[25]:


#mostly 20 rooms in flat cannot be possible, these can be outliers

data[data.bhk >20]


# In[26]:


# 'total_sqft' column currently contains a mix of values, which should not be present in the column. The data in the column 
#should follow a uniform format, whether it's an integer, a float, or a range

#split value which has range and fill it with mean value of it
data['total_sqft'].unique()


# In[27]:


def convert(x):
    temp=x.split('-')
    if len(temp)==2:
        return ((float(temp[0])+float(temp[1]))/2)
    else:
        return x


# In[28]:


#apply convert function in 'total_sqft' column
data['total_sqft']=data['total_sqft'].apply(convert)


# In[29]:


#error above indicates that there are mix values present in the 'total_sqft' column (e.g., 34.46Sq.). First, we need to address
#this issue extract numerical values only
data['total_sqft'] = data['total_sqft'].str.extract('([\d.]+)').astype(float)


# In[30]:


print(type(data['total_sqft'][1]))


# In[31]:


#size column still in multiple data type so we have to convert into price per square feet, for that price in lakh, so we have to convert into lakh, then divide by total sqft.
data['price_per_sqft'] = data['price'] * 100000 / data['total_sqft']


# In[32]:


data.head()


# In[33]:


data['total_sqft'].describe()


# In[34]:


# 'location' column has too many values, which can hinder the use of encoding techniques. Therefore, we need to reduce the number of values in the 'location' column
    
data['location'].value_counts()


# In[35]:


#There could be a chance that the location column value contains white spaces. In the code below, I have used the 'strip' function to remove leading and trailing whitespace.

data['location'] = data['location'].apply(lambda x: x.strip())
location_count = data['location'].value_counts()


# In[36]:


location_count


# In[37]:


#Here, we are finding locations with fewer than 10 values. There are 1053 locations with fewer than 10 values.

location_count_less_then_10=location_count[location_count<=10]
location_count_less_then_10


# In[38]:


#values which are less then 10 in location column we will be replacing it with other

data['location']=data['location'].apply(lambda x: 'other' if x in location_count_less_then_10 else x)

#now check the value counts
data['location'].value_counts()

#now total values become 242


# In[1]:


#As we can see, some flats have an area of only 1 sq. ft which is practically not possible. These are outliers, so we need to remove them.
data.describe()


# In[40]:


#Find flats that have an area less than 300 square feet and then move them

(data['total_sqft']/data['bhk']).describe()


# In[41]:


#as we can see that some flat has 1sqft of area, these are outliers so we have to remove them

data = data[((data['total_sqft']/data['bhk'])>=300)]


# In[42]:


#check describe function to get better idea about monetory value in data
data.describe(percentiles=[0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])

#as we can see in the data outlier are present between 90% - 100% data


# # finding and removing(trimming) the outliers through(IQR method)

# In[43]:


percentile25=data.quantile(0.25)
percentile75=data.quantile(0.75)
IQR=percentile75-percentile25
upper_bounds= percentile75 + IQR*1.5
lower_bounds= percentile25 - IQR*1.5


# In[44]:


new_data = data[
    (data['total_sqft'] >= lower_bounds['total_sqft']) &
    (data['total_sqft'] <= upper_bounds['total_sqft']) &
    (data['bath'] >= lower_bounds['bath']) &
    (data['bath'] <= upper_bounds['bath']) &
    (data['price'] >= lower_bounds['price']) &
    (data['price'] <= upper_bounds['price']) &
    (data['bhk'] >= lower_bounds['bhk']) &
    (data['bhk'] <= upper_bounds['bhk']) &
    (data['price_per_sqft'] >= lower_bounds['price_per_sqft']) &
    (data['price_per_sqft'] <= upper_bounds['price_per_sqft'])
]


# In[45]:


#now we can see that data outlier are removed from all the columns
new_data.describe()


# In[46]:


sns.boxplot(data['price_per_sqft'])


# In[47]:


#droping unnecessary columns from new_data dataframe
new_data.drop(columns=['size','price_per_sqft'], inplace = True)


# In[48]:


new_data.head()


# In[49]:


new_data.to_csv('Cleaned_data_house_price_prediction')


# In[50]:


#In this project, we need to predict a 'price,' which is the default output column. Therefore, we should drop it from the dataset.

X=new_data.drop(columns=['price'])
Y=new_data['price']


# In[51]:


X.shape,Y.shape


# In[75]:


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt


# In[76]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.2, random_state=0)


# In[77]:


print(X_test.shape)


# In[78]:


print(X_train.shape)


# # Apply ML algorithm - Linear 

# In[79]:


columns_trans=make_column_transformer((OneHotEncoder(sparse=False),['location']), remainder='passthrough')


# In[82]:


scaler=StandardScaler()


# In[83]:


lr = LinearRegression()


# In[84]:


pipe= make_pipeline(columns_trans,scaler,lr)


# In[86]:


pipe.fit(X_train,Y_train)


# In[89]:


y_pred_lr=pipe.predict(X_test)


# In[90]:


r2_score(Y_test,y_pred_lr)


# In[ ]:


#here we have get efficiency from model is 67.7%


# # Applying Lasso

# In[92]:


lasso=Lasso()


# In[93]:


pipe=make_pipeline(columns_trans,scaler,lasso)


# In[94]:


pipe.fit(X_train,Y_train)


# In[95]:


Y_pred_lasso=pipe.predict(X_test)
r2_score(Y_test,Y_pred_lasso)


# In[ ]:


#here we have get efficiency from model is 63.7%

