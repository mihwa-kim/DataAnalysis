

import pandas as pd
import os

def create_df(filename):
    PATH = '/Users/gimmihwa/Desktop/DATA/Data Sets/Locations near EV chargers/Community Center/'
    df = pd.read_csv(PATH+filename, error_bad_lines=False, warn_bad_lines=True)

    # change all values in title to Community center
    df['title'] = 'Community center'

    df_store_location = df[['title', 'lon', 'lat']]
    df_store_location = df_store_location.fillna('0')

    # move the index to the first column
    df_store_location = df_store_location.reset_index()
    

    df_store_location.rename(columns={"index": "Name"}, inplace=True)
    
    return df_store_location

def generate_merge_by_path(PATH):
    file_list, csv_list = os.listdir(PATH), list()
    dataframes = []  # vacant list
    for file in file_list:
        if file.split('.')[-1] == 'csv':
            csv_list.append(file)
    csv_list.sort()
    
    for file in csv_list:
        doc = create_df(file)
        dataframes.append(doc)  # append dataframe to the list
    
    # all dataframe concat connect
    final_doc = pd.concat(dataframes, ignore_index=True)
    final_doc = final_doc.fillna(0)
    return final_doc

PATH_1 = '/Users/gimmihwa/Desktop/DATA/Data Sets/Locations near EV chargers/Community Center/'
df_community = generate_merge_by_path(PATH_1)

df_community

df_community.to_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/Community Center_merged.csv')


# In[62]:


# How to merge if I have None CSV file ( shopping and restaurant_merged)

def create_df(filename):
    PATH = '/Users/gimmihwa/Desktop/DATA/Data Sets/Locations near EV chargers/shopping/'
    file_path = PATH + filename

    # confirm file details
    if os.path.getsize(file_path) == 0:
        print(f"Skipping empty file: {filename}")
        return None

    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
    
    df['title'] = 'shopping'

    df_store_location = df[['title', 'lon', 'lat']]
    df_store_location = df_store_location.fillna('0')
    
    df_store_location = df_store_location.reset_index()
    
    df_store_location.rename(columns={"index": "Name"}, inplace=True)
    
    return df_store_location


def generate_merge_by_path(PATH):
    file_list, csv_list = os.listdir(PATH), list()
    dataframes = []
    for file in file_list:
        if file.split('.')[-1] == 'csv':
            csv_list.append(file)
    csv_list.sort()
    
    for file in csv_list:
        doc = create_df(file)
        if doc is not None:
            dataframes.append(doc)
    
    final_doc = pd.concat(dataframes, ignore_index=True)
    final_doc = final_doc.fillna(0)
    return final_doc

PATH_1 = '/Users/gimmihwa/Desktop/DATA/Data Sets/Locations near EV chargers/shopping/'
df_community = generate_merge_by_path(PATH_1)

df_community

df_community.to_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/shopping_merged.csv')


# In[63]:
# merge all the customer services csv( shopping, community, restaurant)

shopping = pd.read_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/shopping_merged.csv')
community = pd.read_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/community_merged.csv')
restaurant = pd.read_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/restaurant_merged.csv')

dataframes = [shopping,community,restaurant]

final_doc = pd.concat(dataframes, ignore_index=True)

final_doc.to_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/final_merged.csv')


# ### Total_vehicles_registered_in_the_ACT ) registered count by vehicle type

# In[76]:


vehicle = pd.read_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/Total_vehicles_registered_in_the_ACT.csv')
vehicle.head()


# In[77]:


vehicle.info()


# In[78]:


vehicle['Date'] = pd.to_datetime(vehicle['Date'])


# In[79]:


vehicle_grouping = vehicle.groupby(['Date','Vehicle body type']).sum()


# In[91]:


vehicle_grouping = vehicle.reset_index()
vehicle_grouping
pivoted = vehicle_grouping.pivot_table(index='Vehicle body type', columns='Date', values='Count', aggfunc='sum')
pivoted.fillna(0)
vehicle_updated = pivoted
vehicle_updated.to_csv('/Users/gimmihwa/Desktop/DATA/Data Sets/merged file/vehicle_updated.csv')



