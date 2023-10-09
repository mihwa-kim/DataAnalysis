#!/usr/bin/env python
# coding: utf-8

# ## Column Description
# * Countries: This column contains the names of countries. It is challenging to differentiate between African and South American countries as the continent information is not available, but it seems to be comprised only of developing countries. GEF Grant: This represents the grant for climate system development. Cofinancing: This refers to joint financing. Approval FY: This is the year the project was approved. The 'Status' is set as the dependent variable (y).
# 
# ## TYPE:
# * It is observed that there are very few cancelled projects. The projects have been primarily executed as full-size projects, enabling activity, medium-size projects, and PFDs, in that order. Excluding PFDs, full-size projects have the highest completion rate, even higher than the combined rate of the other two types. The ratio of approved projects to completed projects is higher for full-size projects and about 60% for enabling activities.
# 
# ## Agency:
# * Projects completed through the United Nations Development Programme are significantly higher in number. The projects are predominantly conducted through the top 5 agencies. The World Bank has a higher ratio of cancelled to completed projects, with the highest number of cancelled projects through this agency.
# 
# ## Approved FY:
# * Until 2014, there was a trend of increasing project counts. However, there was a sharp decline afterward. In 2022, there was a brief significant increase, but it sharply decreased again in 2023. Examining the status of these projects, it is noted that completed projects significantly decreased around 2014, while approved projects significantly increased but have been sharply decreasing in 2023 (likely due to the unavailability of complete data). Cancelled projects do not reveal any significant insights.
# 
# ## Funding Source (indexed field) & Non-Grant Instrument & Capacity-building & GEF Period:
# * Over 80% of the projects use the GEF Trust Fund source. For projects where Grant Instrument is 'Yes' and Capacity-building Initiative for Transparency is 'No', there is a higher probability of project execution. For GEF periods 5 to 7, there is a distinct increase in project approvals. GEF periods 1 and 4, 3, 2, 5 have a higher completion rate. However, period 4 also has the highest cancellation rate.

# In[2]:


import pandas as pd


# In[5]:


df = pd.read_csv('projects (1).csv')
df.head()


# In[3]:


df.shape


# In[6]:


df['Title'].unique()


# In[31]:


df.info()


# In[43]:


df['GEF Grant'] = df['GEF Grant'].str.replace(',', '').fillna(0).astype(int)


# In[44]:


df['Cofinancing'] = df['Cofinancing'].str.replace(',', '').fillna(0).astype(int)


# In[45]:


# Assuming 'Approval FY' is in a recognizable datetime format
# df['Approval FY'] = pd.to_datetime(df['Approval FY']).dt.year
df['Approval FY'] = df['Approval FY'].fillna(0).astype(int)


# In[47]:


df['GEF Period'].unique()


# In[48]:


df.describe()


# ### type

# In[59]:


import pandas as pd
import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline(connected=True)


# In[66]:


# Cancelled                       214
# Completed                      3095
# Concept Approved                311
# Concept Proposed                 17
# Project Approved               2145
# Received by GEF Secretariat      17

completed = df[df['Status'] == 'Completed']['Type'] 
cancelled = df[df['Status'] == 'Cancelled']['Type']
p_approved = df[df['Status'] == 'Project Approved']['Type']
c_approved = df[df['Status'] == 'Concept Approved']['Type']
c_proposed = df[df['Status'] == 'Concept Proposed']['Type'] 
receive = df[df['Status'] == 'Received by GEF Secretariat']['Type'] 
temp = pd.concat([completed, cancelled,p_approved,c_approved,c_proposed,receive], axis=1, keys=['Completed', 'Cancelled','Project Approved','Concept Approved',
                                                     'Concept Proposed', 'Received by GEF Secretariat'])


# In[67]:


temp.iplot(kind='histogram')


# 캔슬된 프로젝트는 거의 없다는 것을 알수있다. 
# full-size project, enabling activity, medium-size project, PFD 순으로 많이 실행되었다.
# PFD를 제외하고 완성된 프로젝트가 가장 많으나 그 비율이 풀타임 프로젝트에서 나머지 두개를 합한 것만큼 많은 걸 알수있다.
# 완성된 프로젝트 대비 프로젝트 승인 비율이 풀 사이즈 프로젝트에서 많은 편, enabling activity에서는 완성된 프로젝트가 승인 대비 60%로 많다.

# ### Agencies

# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns

# 데이터프레임에서 그룹화하고 정렬합니다.
agency_counts = df.groupby('Agencies')['ID'].count().sort_values(ascending=False)

# 바 차트를 그립니다.
# plt.figure(figsize=(10,8))
# sns.barplot(x=agency_counts.values, y=agency_counts.index)
# plt.xlabel('Number of IDs')
# plt.ylabel('Agencies')
# plt.title('Number of IDs by Agency')
# plt.show()

# 상위 10개 기관만 보여줍니다.
top_n = 20
top_agencies = agency_counts.head(top_n)

plt.figure(figsize=(10,6))
sns.barplot(x=top_agencies.values, y=top_agencies.index)
plt.xlabel('Number of IDs')
plt.ylabel('Agencies')
plt.title('Number of IDs by Agency (Top 10)')
plt.show()



# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns

# 상위 20개 기관만 선택합니다.
top_n = 20
top_agencies = df['Agencies'].value_counts().head(top_n).index

# 상위 20개 기관의 데이터만 필터링합니다.
filtered_df = df[df['Agencies'].isin(top_agencies)]

# 그래프를 그립니다.
plt.figure(figsize=(15,10))
sns.countplot(y='Agencies', hue='Status', data=filtered_df, order=top_agencies)
plt.xlabel('Count')
plt.ylabel('Agencies')
plt.title('Status Frequency for each Agency (Top 20)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# United Nations Development Programme 통해  completed 된 프로젝트 수 압도적으로 많음
# Top 5 에이전시 중심으로 프로젝트 진행되는 편임
# The world bank 는 completed 대비 cancelled가 높은 편이다.( 해당 기관통해 취소된 프로젝트 가장 많음)

# In[77]:


df.groupby('Approval FY')['ID'].count()


# ### Approval FY

# In[82]:


import matplotlib.pyplot as plt

# 각 'Approval FY'별로 'ID'의 개수를 계산합니다.
approval_fy_counts = df.groupby('Approval FY')['ID'].count()

# 라인 그래프로 시각화합니다.
plt.figure(figsize=(12,6))
approval_fy_counts.plot(kind='line', marker='o', xlim=(1990, 2023)) # x축의 범위를 1990년부터 2023년까지로 설정합니다.
plt.xlabel('Approval FY')
plt.ylabel('Number of IDs')
plt.title('Number of IDs by Approval FY')
plt.grid(True)
plt.xticks(range(1990, 2024, 5))  # x축의 눈금을 5년 간격으로 설정합니다.
plt.show()


# 2014년까지 프로젝트 갯수 증가하는 경향을 보였으나 이후 급격히 감소하였고 2022년에 잠깐 크게 올랐으나 2023년 되어 다시 급감하는 추세입니다.
# 해당 프로젝트들의 상태 정보를 확인해보았습니다.
# 완료된 프로젝트는 2014 년 기점으로 큰 폭으로 감소하였고, 승인된 프로젝트는 해당 기점으로 크게 증가하였으나 2023년되어 급감하는 모습(아직 전체 데이터 확보 안되어 그럴 가능성 큼)
# 취소된 프로젝트는 특별한 내용을 확인하기 어려워 보임

# In[89]:


grouped_counts = df.groupby(['Approval FY', 'Status']).size()
completed_counts = grouped_counts.xs('Completed', level=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
completed_counts.plot(kind='line', marker='o', xlim=(1990, 2023))
plt.xlabel('Approval FY')
plt.ylabel('Number of Completed Projects')
plt.title('Number of Completed Projects by Approval FY')
plt.grid(True)
plt.xticks(range(1990, 2024, 5))
plt.show()


# In[90]:


grouped_counts = df.groupby(['Approval FY', 'Status']).size()
cancelled_counts = grouped_counts.xs('Cancelled', level=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
cancelled_counts.plot(kind='line', marker='o', xlim=(1990, 2023))
plt.xlabel('Approval FY')
plt.ylabel('Number of Cancelled Projects')
plt.title('Number of Cancelled Projects by Approval FY')
plt.grid(True)
plt.xticks(range(1990, 2024, 5))
plt.show()


# In[92]:


status_list = df['Status'].unique()


# In[99]:


type(status_list)


# In[103]:


import numpy as np

grouped_counts = df.groupby(['Approval FY', 'Status']).size()

for data in status_list:
    if pd.isna(data):  # nan 값인지 확인
        continue  # nan 값이면 건너뛰기
    
    cancelled_counts = grouped_counts.xs(data, level=1)

    plt.figure(figsize=(12,6))
    cancelled_counts.plot(kind='line', marker='o', xlim=(1990, 2023))
    plt.xlabel('Approval FY')
    plt.ylabel('Number of '+data+' Projects')
    plt.title('Number of '+ data+ ' Projects by Approval FY')
    plt.grid(True)
    plt.xticks(range(1990, 2024, 5))
    plt.show()


# In[104]:


df.info()


# GEF Trust Fund  소스 80% 이상 사용함

# In[107]:


df.groupby('Funding Source (indexed field)')['ID'].count()


# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns

# # 상위 20개 기관만 선택합니다.
# top_n = 20
# top_agencies = df['Agencies'].value_counts().head(top_n).index

# # 상위 20개 기관의 데이터만 필터링합니다.
# filtered_df = df[df['Agencies'].isin(top_agencies)]

# 그래프를 그립니다.
plt.figure(figsize=(15,10))
sns.countplot(y='Funding Source (indexed field)', hue='Status', data=df)
plt.xlabel('Count')
plt.ylabel('Funding Source')
plt.title('Status Frequency for each Funding Source')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[109]:


df.groupby('Non-Grant Instrument (indexed field)')['ID'].count()


# In[110]:


import matplotlib.pyplot as plt
import seaborn as sns

# # 상위 20개 기관만 선택합니다.
# top_n = 20
# top_agencies = df['Agencies'].value_counts().head(top_n).index

# # 상위 20개 기관의 데이터만 필터링합니다.
# filtered_df = df[df['Agencies'].isin(top_agencies)]

# 그래프를 그립니다.
plt.figure(figsize=(15,10))
sns.countplot(y='Non-Grant Instrument (indexed field)', hue='Status', data=df)
plt.xlabel('Count')
plt.ylabel('Non-Grant Instrument')
plt.title('Status Frequency for Non-Grant Instrument')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[111]:


print(df.groupby('Capacity-building Initiative for Transparency')['ID'].count())

import matplotlib.pyplot as plt
import seaborn as sns

# # 상위 20개 기관만 선택합니다.
# top_n = 20
# top_agencies = df['Agencies'].value_counts().head(top_n).index

# # 상위 20개 기관의 데이터만 필터링합니다.
# filtered_df = df[df['Agencies'].isin(top_agencies)]

# 그래프를 그립니다.
plt.figure(figsize=(15,10))
sns.countplot(y='Capacity-building Initiative for Transparency', hue='Status', data=df)
plt.xlabel('Count')
plt.ylabel('Capacity-building Initiative for Transparency')
plt.title('Status Frequency for Capacity-building Initiative for Transparency')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



# In[113]:


print(df.groupby('GEF Period')['ID'].count())

import matplotlib.pyplot as plt
import seaborn as sns

# # 상위 20개 기관만 선택합니다.
# top_n = 20
# top_agencies = df['Agencies'].value_counts().head(top_n).index

# # 상위 20개 기관의 데이터만 필터링합니다.
# filtered_df = df[df['Agencies'].isin(top_agencies)]

# 그래프를 그립니다.
plt.figure(figsize=(15,10))
sns.countplot(y='GEF Period', hue='Status', data=df)
plt.xlabel('Count')
plt.ylabel('GEF Period')
plt.title('Status Frequency for GEF Period')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()



# 

# In[ ]:





# In[ ]:





# In[53]:


df.groupby('Status')['Countries'].count()


# In[54]:


df.groupby(['Approval FY','Status'])['Countries'].count()


# In[1]:


df.info()


# In[ ]:




