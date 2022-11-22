#!/usr/bin/env python
# coding: utf-8

# ## DSC 180AB Data Science Capstone
# ### Replication Project

# Team Members:

# ### Table of Contents
# To return to the table of contents, click on the number at any major section heading.

# [1. Introduction](#1.-Introduction)
# 
# [2. Exploratory Data Analysis](#2.-Exploratory-Data-Analysis)
# 
# [3. Model Development](#3.-Model-Development)
# 
# [4. Model Evaluation](#4.-Model-Evaluation)
# 
# [5. Bias Mitigation](#5.-Bias-Mitigation)
# 
# [6. Results Summary](#6.-Results-Summary)
# 
# [7. Explainability](#7.-Explainability)
# 
# [8. Conclusion & Discussion](#8.-Conclusion-&-Discussion)
# 

# ## This tutorial demonstrates classification model learning with bias mitigation as a part of a Care Management use case using Medical Expenditure data.

# The notebook demonstrates how the AIF 360 toolkit can be used to detect and reduce bias when learning classifiers using a variety of fairness metrics and algorithms . It also demonstrates how explanations can be generated for predictions made by models learnt with the toolkit using LIME.
# 
# * Classifiers are built using Logistic Regression as well as Random Forests.
# * Bias detection is demonstrated using several metrics, including disparate impact, average odds difference, statistical parity difference, equal opportunity difference, and Theil index.
# * Bias alleviation is explored via a variety of methods, including reweighing (pre-processing algorithm), prejudice remover (in-processing algorithm), and disparate impact remover (pre-processing technique).
# * Data from the [Medical Expenditure Panel Survey](https://meps.ahrq.gov/mepsweb/) is used in this tutorial.
# 
# 
# The Medical Expenditure Panel Survey (MEPS) provides nationally representative estimates of health expenditure, utilization, payment sources, health status, and health insurance coverage among the noninstitutionalized U.S. population. These government-produced data sets examine how people use the US healthcare system.
# 
# MEPS is administered by the Agency for Healthcare Research and Quality (AHRQ) and is divided into three components: 
# * Household
# * Insurance/Employer, and 
# * Medical Provider. 
# 
# These components provide comprehensive national estimates of health care use and payment by individuals, families, and any other demographic group of interest.

# ### [1.](#Table-of-Contents) Introduction

# The [AI Fairness 360 toolkit](https://github.com/Trusted-AI/AIF360) is an extensible open-source library containing techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle. AI Fairness 360 package is available in both Python and R. Documentation is available [here](https://aif360.readthedocs.io/en/v0.2.3/index.html)
# 
# The AI Fairness 360 package includes: 
# - a comprehensive set of metrics for datasets and models to test for biases,
# - explanations for these metrics, and
# - algorithms to mitigate bias in datasets and models
# It is designed to translate algorithmic research from the lab into the actual practice of domains as wide-ranging as finance, human capital management, healthcare, and education

# #### 1.1 Use Case
# 
# **In order to demonstrate how AIF360 can be used to detect and mitigate bias in classfier models, we adopt the following use case:**
# 
# * Data scientist develops a 'fair' healthcare utilization scoring model with respect to defined protected classes. Fairness may be dictated by legal or government regulations, such as a requirement that additional care decisions be not predicated on factors such as race of the patient.
# * Developer takes the model AND performance characteristics / specs of the model (e.g. accuracy, fairness tests, etc. basically the model factsheet) and deploys the model in an enterprise app that prioritizes cases for care management.
# * The app is put into production and starts scoring people and making recommendations. 
# * Explanations are generated for each recommendation
# * Both recommendations and associated explanations are given to nurses as a part of the care management process. The nurses can evaluate the recommendations for quality and correctness and provide feedback.
# * Nurse feedback as well as analysis of usage data with respect to specs of the model w.r.t accuracy and fairness is communicated to AI Ops specialist and LOB user periodically.
# * When significant drift in model specs relative to the model factsheet is observed, the model is sent back for retraining.

# #### 1.2 Data
# Released as an ASCII file (with related SAS, SPSS, and STATA programming statements) and a SAS transport dataset, this public use file provides information collected on a nationally representative sample of the civilian noninstitutionalized population of the United States for calendar year 2015. This file consists of MEPS survey data obtained in Rounds 3, 4, and 5 of Panel 19 and Rounds 1, 2, and 3 of Panel 20 (i.e., the rounds for the MEPS panels covering calendar year 2015) and consolidates all of the final 2015 person-level variables onto one file. This file contains the following variables previously released on HC-174: survey administration, language of interview variable, demographics, parent identifiers, health status, disability days variables, access to care, employment, quality of care, patient satisfaction, health insurance, and use variables. The HC-181 file also includes these variables: income variables and expenditure variables.
# 
# The specific data used is the [2015 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) as well as the [2016 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192).
# 
# * The 2015 file contains data from rounds 3,4,5 of panel 19 (2014) and rounds 1,2,3 of panel 20 (2015). 
# * The 2016 file contains data from rounds 3,4,5 of panel 20 (2015) and rounds 1,2,3 of panel 21 (2016).
# 
# In this example, three datasets were constructed: one from panel 19, round 5 (used for learning models), one from panel 20, round 3 (used for deployment/testing of model - steps); the other from panel 21, round 3 (used for re-training and deployment/testing of updated model).

# #### 1.3 Methodology 
# 
# For each dataset, the sensitive attribute is 'RACE' constructed as follows: 'Whites' (privileged class) defined by the features RACEV2X = 1 (White) and HISPANX = 2 (non Hispanic); 'Non-Whites' that included everyone else.  
# 
# * Along with race as the sensitive feature, other features used for modeling include demographics  (such as age, gender, active duty status), physical/mental health assessments, diagnosis codes (such as history of diagnosis of cancer, or diabetes), and limitations (such as cognitive or hearing or vision limitation).
# * To measure utilization, a composite feature, 'UTILIZATION', was created to measure the total number of trips requiring some sort of medical care by summing up the following features: OBTOTV15(16), the number of office based visits;  OPTOTV15(16), the number of outpatient visits; ERTOT15(16), the number of ER visits;  IPNGTD15(16), the number of inpatient nights, and  + HHTOTD16, the number of home health visits.
# * The model classification task is to predict whether a person would have 'high' utilization (defined as UTILIZATION >= 10, roughly the average utilization for the considered population). High utilization respondents constituted around 17% of each dataset.
# * To simulate the scenario, each dataset is split into 3 parts: a train, a validation, and a test/deployment part.
# 
# **We assume that the model is initially built and tuned using the 2015 Panel 19 train/test data**
# * It is then put into practice and used to score people to identify potential candidates for care management. 
# * Initial deployment is simulated to 2015 Panel 20 deployment data. 
# * To show change in performance and/or fairness over time, the 2016 Panel 21 deployment data is used. 
# * Finally, if drift is observed, the 2015 train/validation data is used to learn a new model and evaluated again on the 2016 deployment data

# ### 1.4 Insert writeup of overall replication project goals and big picture thinking (2-3 paragraphs).  
# * Why do we care about this? 
# * What would the benefit of predicting utilization be? 
# * What might occur if there are errors?
# * Who are the affected parties and stakeholders?
# * Other thoughts?

# **Write up here:**
# 
# Healthcare utilization is a measurement of a patient’s total number of trips to some sort of medical care, a trait commonly associated with higher health care costs. 
# 
# The reason we care about predicting healthcare utilization is because by ensuring that various groups of stakeholders can receive the appropriate standard of care, we can potentially help enable a healthcare system that is able to give the right amount of care to the right patients who require it. Ultimately the US healthcare system is commonly considered by both its local populace and the international community to be highly inefficient, so this predictive system is a step in the right direction for perhaps the nation as a whole.
# 
# Being able to predict healthcare utilization would be of great benefit to many stakeholders traditionally associated with health insurance. Insurance providers would be able to create or tailor existing plans to better target groups in need based on their projected utilization of services. The resulting, well-working recommendations would be able to block users into utilization groups to maximize their likelihood of acquiring insurance best suited to their healthcare utilization needs. In turn, widespread adoption of more utilization accurate healthcare plans would lead to higher insurance utilization for consumers and support from hospitals, providing health benefits to the populace. Additionally, having a better recognition of what factors cause greater utilization could be of benefit to physicians looking to recommend plans suitable for their patients. 
# 
# Errors in utilization prediction would lead to mismatched insurance plans for users based on their utilization. Recommending users an insurance plan with insufficient coverage for things like inpatient services may lead to a significant increase in out-of-pocket financial burden. Similarly, burdening customers with increased insurance costs relative to their needs would be wasteful for the consumer. 

# ---
# End of Introduction

# ### [2.](#Table-of-Contents) Exploratory Data Analysis (EDA)
# 

# The specific data used is the [2015 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) as well as the [2016 Full Year Consolidated Data File](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192).
# 
# * The 2015 file contains data from rounds 3,4,5 of panel 19 (2014) and rounds 1,2,3 of panel 20 (2015). 
# * The 2016 file contains data from rounds 3,4,5 of panel 20 (2015) and rounds 1,2,3 of panel 21 (2016).
# 
# In this example, three datasets were constructed: one from panel 19, round 5 (used for learning models), one from panel 20, round 3 (used for deployment/testing of model - steps); the other from panel 21, round 3 (used for re-training and deployment/testing of updated model).
# 
# See the corresponding [Codebook](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) for information on variables.
# 
# ##### Key MEPS dataset features include:
# * **Utilization**: To measure utilization, a composite feature, 'UTILIZATION', was created to measure the total number of trips requiring some sort of medical care by summing up the following features: OBTOTV15(16), the number of office based visits;  OPTOTV15(16), the number of outpatient visits; ERTOT15(16), the number of ER visits;  IPNGTD15(16), the number of inpatient nights, and  + HHTOTD16, the number of home health visits.
# * The model classification task is to predict whether a person would have **'high'** utilization (defined as UTILIZATION >= 10, roughly the average utilization for the considered population). High utilization respondents constituted around 17% of each dataset.

# #### 2.0 Pre-processing Scripts (for each Panel)
# 
# There is currently minimal EDA for this tutorial within IBM AIF360 Medical Expenditure Tutorial. Therefore, we have adapted  utility scripts from IBM AIF360 Tutorial for ease of understanding for how datasets were pre-processed. These will be used primarily for EDA purposes. We will utilize IBM's tutorial for the remainder of the project. We have utilized Pandas for this portion of the project. 
# 
# **Note:** these pre-processing script below are run for each data file, and then filtered for each panel. This was done in order to match subsequent portions of the tutorial, and how train/test/validation datasets were split.

# #### 2.1 Get and Load Dataset, Apply Pre-processing

# **Before Proceeding Ensure You Have:**
# * Forked the AIF360 R=repository and cloned locally to your disk or virtual machine
# * Downloaded the `h181.csv` and `h192.csv` data files uploaded [here](https://www.kaggle.com/datasets/nanrahman/mepsdata)
# * Place the `h181.csv` and `h192.csv` in a folder you can access (we placed it in `../aif360/data/raw/meps/` of our forked AIF360 repository)
# * For EDA we only focus on `h181.csv` 

# In[1]:


# Imports
import sys

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
import pandas as pd
import seaborn as sns

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")


# In[2]:


raw_181 = pd.read_csv('test/testdata/h181.csv')


# #### Apply pre-processing scripts

# In[3]:


default_mappings = {
    'label_maps': [{1.0: '>= 10 Visits', 0.0: '< 10 Visits'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-White'}]}

def default_preprocessing19(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 19
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """
    df = df.copy()
    def race(row):
        if ((row['HISPANX'] == 2) & (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE#return 'White'
            return 'White'
        return 'Non-White'

    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns = {'RACEV2X' : 'RACE'})
    
    df = df[df['PANEL'] == 19]

    # RENAME COLUMNS
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1

    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

    def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df['TOTEXP15'] < 10.0
    df.loc[lessE,'TOTEXP15'] = 0.0
    moreE = df['TOTEXP15'] >= 10.0
    df.loc[moreE,'TOTEXP15'] = 1.0

    df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
    return df


# In[4]:


def default_preprocessing20(df):
    """
    1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
      and 'non-White' otherwise
    2. Restrict to Panel 20
    3. RENAME all columns that are PANEL/ROUND SPECIFIC
    4. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
    5. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    """
    df = df.copy()
    def race(row):
        if ((row['HISPANX'] == 2) & (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
            return 'White'
        return 'Non-White'

    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns = {'RACEV2X' : 'RACE'})

    df = df[df['PANEL'] == 20]

    # RENAME COLUMNS
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1

    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1

    def utilization(row):
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df['TOTEXP15'] < 10.0
    df.loc[lessE,'TOTEXP15'] = 0.0
    moreE = df['TOTEXP15'] >= 10.0
    df.loc[moreE,'TOTEXP15'] = 1.0

    df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
    return df


# #### Taken from pre-processing scripts to retain same columns used in model development for tutorial

# In[5]:


label_name='UTILIZATION'
favorable_classes=[1.0]
protected_attribute_names=['RACE']
privileged_classes=[['White']]
instance_weights_name='PERWT15F'
categorical_features=['REGION','SEX','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42', 'PHQ242',
                                 'EMPST','POVCAT','INSCOV']

features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                 'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']
features_to_drop=[]
na_values=[]
# custom_preprocessing=default_preprocessing <- don't need this yet for EDA
metadata=default_mappings


# We encourage you to search through the repository and take a look at these scripts, 
# they can be found in `../aif360/dataset/` in your forked AIF360 repository:
# * AIF360/aif360/datasets/meps_dataset_panel19_fy2015.py
# * AIF360/aif360/datasets/meps_dataset_panel20_fy2015.py
# 
# To Explore the `Utilization` and `RACE` features, and the variables used to impute these features:
# * See the corresponding [HC 181 Codebook](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181) for information on variables.

# In[6]:


df_panel_19 = default_preprocessing19(raw_181)
df_panel_19_reduced = df_panel_19[features_to_keep]


# In[7]:


df_panel_20 = default_preprocessing20(raw_181)
df_panel_20_reduced = df_panel_20[features_to_keep]


# In[8]:


#### END OF PRE-PROCRESSING ####


# #### 2.2 Data shape and features

# In[9]:


df_panel_19_reduced


# In[10]:


df_panel_20_reduced


# In[11]:


# Identify feature names
# Check for categorical features

feature_names = np.union1d(df_panel_20_reduced.columns, df_panel_19_reduced.columns)
feature_names = feature_names[feature_names != "UTILIZATION"]

# existing 'categorical' features taken from the pre-procesing scripts above
categorical = np.intersect1d(feature_names, categorical_features)
non_categorical = np.setdiff1d(feature_names, categorical_features)

categorical = np.append(categorical, protected_attribute_names)
non_categorical = non_categorical[non_categorical != "RACE"]


# for feature descriptions and names, see dictionary
categorical, non_categorical


# In[12]:


df_panel_19_reduced["RACE"].value_counts(dropna =False)  # not sure how they got a "None" into the dataset? Should this be "White"?


# In[13]:


df_panel_20_reduced["RACE"].value_counts(dropna =False)


# In[14]:


# Summary statistics
# Shapes
df_panel_19_reduced[non_categorical].describe()


# In[15]:


df_panel_20_reduced[non_categorical].describe()


# In[16]:


df_panel_19_reduced.shape, df_panel_20_reduced.shape


# #### 2.3 Outlier Detection and Handling

# For null handling we will do the following:
# - We will clean the RACE column to be what it needs to be.
#     - This was done by fixing the preprocessing script.
# - We will leave -1 and -9 values in categorical columns as is as they can be treated as other categories.
# - We will use sampling imputation for imputing -1 and -9 values in quantitative columns as shown below.

# In[17]:


#Before handling
df_panel_19_reduced[non_categorical]


# In[18]:


# Null handling
non_categorical_data = df_panel_19_reduced[non_categorical]
    
for col in non_categorical_data.columns:
    my_filter = non_categorical_data[col] < 0
    sample_from = non_categorical_data.loc[~my_filter,col]
    my_sample = sample_from.sample(len(non_categorical_data.loc[my_filter,col]),replace=True)
    non_categorical_data.loc[my_filter,col] = my_sample.values

df_panel_19_reduced[non_categorical] = non_categorical_data


# In[19]:


df_panel_19_reduced[non_categorical]


# #### 2.4 Correlation Analysis

# In[20]:


# Preliminary data visualizations
# Preliminary data visualizations
race = df_panel_19_reduced['RACE']
students = df_panel_19_reduced[df_panel_19_reduced['FTSTU'] == 1]
smokers = df_panel_19_reduced[df_panel_19_reduced['ADSMOK42'] == 1]
nonsmokers = df_panel_19_reduced[df_panel_19_reduced['ADSMOK42'] == 2]

high_bp = df_panel_19_reduced[df_panel_19_reduced['HIBPDX'] == 1]
arth = df_panel_19_reduced[df_panel_19_reduced['ARTHDX'] == 1]
high_ch = df_panel_19_reduced[df_panel_19_reduced['CHOLDX'] == 1]
coronary = df_panel_19_reduced[df_panel_19_reduced['CHDDX'] == 1]


#Non-Whites with Cancer
df_panel_19_reduced[race == 'Non-White']['CANCERDX'].value_counts().rename({-1: 'Inapplicable', 1: 'Yes', 2: 'No'}) .plot(kind='barh', title='Non-White Individuals with Cancer')
plt.show()

#White with Cancer
df_panel_19_reduced[race != 'Non-White']['CANCERDX'].value_counts().rename({-1: 'Inapplicable', 1: 'Yes', 2: 'No'}).plot(kind='barh', title='White Individuals with Cancer')
plt.show()

#Non-Whites with Health Insurance
df_panel_19_reduced[race == 'Non-White']['INSCOV'].value_counts().rename({1: 'Private', 2: 'Public', 3: 'Uninsured'}) .plot(kind='barh', title='Non-White Individuals Health Insurance Status')
plt.show()

#White with Health Insurance
df_panel_19_reduced[race != 'Non-White']['INSCOV'].value_counts().rename({1: 'Private', 2: 'Public', 3: 'Uninsured'}).plot(kind='barh', title='White Individuals Health Insurance Status')
plt.show()

#Smokers with Cancer
smokers['CANCERDX'].value_counts().rename({-1: 'Inapplicable', 1: 'Yes', 2: 'No'}).plot(kind='barh', title='Smokers Cancer Status')
plt.show()

#Non-Smokers with Cancer
nonsmokers['CANCERDX'].value_counts().rename({-1: 'Inapplicable', 1: 'Yes', 2: 'No'}).plot(kind='barh', title='Non-Smokers Cancer Status')
plt.show()

#Student Mental Health
students['MNHLTH'].value_counts().rename({1: 'Inapplicable', -1: 'Execellent', 2: 'Very Good', 3: 'Good',4: 'Fair',5:'Poor'}).plot(kind='barh', title='Full Time Students Mental Health')
plt.show()

#High Blood Pressure by Age
high_bp.groupby(by="AGE").size().plot(kind='line', title='High Blood Pressure by Age')
plt.show()

#Arthritis by Age
arth.groupby(by="AGE").size().plot(kind='line', title='Arthritis by Age')
plt.show()

#High Cholesterol Pressure by Age
high_ch.groupby(by="AGE").size().plot(kind='line', title='High Cholesterol Pressure by Age')
plt.show()

#Coronary Heart Disease by Age
coronary.groupby(by="AGE").size().plot(kind='line', title='Coronary Heart Disease by Age')
plt.show()


# In[21]:


# Correlation plot 1
corr_plot_19 = pd.plotting.scatter_matrix(df_panel_19_reduced[np.append(non_categorical, 'UTILIZATION') ])

# Correlation plot 2
corr_plot_20 = pd.plotting.scatter_matrix(df_panel_20_reduced[np.append(non_categorical, 'UTILIZATION') ])


# #### 2.5 Other analysis

# In[22]:


# Since the main feature we're predicting is utilization and the correlation plots above are kind of hard to read, let's 
# figure out what features are most correlated with utilization:
corr_19 = df_panel_19_reduced[np.append(non_categorical, 'UTILIZATION')].corr().abs()
corr_19['UTILIZATION'].sort_values(ascending=False)[1:11]


# In[23]:


# Next let's figure out how much of our population has a 1 v 0 in UTILIZATION:
prop_1 = df_panel_19_reduced['UTILIZATION'].mean()
prop_0 = 1 - prop_1
print('1: ', prop_1)
print('0: ', prop_0)


# In[24]:


# But this isn't exactly the best way to go about determining the correlation between a continuous and dichotomous variable
# Instead I'll be using the point biserial correlation in the following cell to figure out what the correlations are:
import scipy.stats as stats

for i in non_categorical:
    print(f"The point-biserial correlation between UTILIZATION and {i} is: ", 
          stats.pointbiserialr(df_panel_19_reduced[i], df_panel_19_reduced['UTILIZATION'])[0])


# -----
# End of Exploratory Data Analysis

# 
# ### End of Replication Part 01 -  EDA
# 

# -----
# # Start of Replication Part 02 -  Model Development, and Fairness Evaluation
# 
# ## There are **two** components to `Replication Project Part #02`
# 1. Training models without de-biasing, using IBM's tutorial
# 2. Training models without de-biasing, using your own model development techniques including (1) Feature Selection, (2) Encoding, (3) Binning Features, and other items 
# 
# #### We will now return to IBM AIF360's [Medical Expenditure Tutorial](https://nbviewer.org/github/IBM/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb) 
# _*Note that it is primarily Scikit-learn based_
# 
# * A reminder, you will need to fork [AIF360's repository](https://github.com/Trusted-AI/AIF360) into your own GitHub and access the notebook locally or via your method of choice
# * AIF360's Repository can be found under: `AIF360`/`Examples`/tutorial_medical_expenditure.ipynb
# * Ensure you have your `aif360` environment turned and activated using a miniconda prompt
# * Use Jupyter Labs
# * Refer to [Week 03](https://nanrahman.github.io/capstone-responsible-ai/weeks/03-Replication-Part-00/) content on the course Website to access the `Quickstart Guide`
# 
# 
# 
# 
# 

# ### [3.](#Table-of-Contents) Model Development without Debiasing 
# 

# First, load all necessary packages

# In[25]:


import sys
sys.path.insert(0, '../')

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.sklearn import metrics

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# LIME
from aif360.datasets.lime_encoder import LimeEncoder
import lime
from lime.lime_tabular import LimeTabularExplainer

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score

np.random.seed(1)


# ### 3.1. Load data & create splits for learning/validating/testing model

# In[ ]:





# ### 3.2. Learning a Logistic Regression (LR) classifier on original data

# ### 3.3. Learning a Random Forest (RF) classifier on original data

# ### Section 3 Write Up here
# 
# ### Part-01: For **both** the logistic regression and random forest classifiers learned on the original data, please include explain the results of your fairness metrics. For _each_ metric result briefly describe what this value means in 1-2 sentences (is it fair, is it not fair? Why?)
# 
# _Logistic Regression Fairness Metrics_
#    * Threshold corresponding to best balanced accuracy: 0.1900
#    * Best balanced accuracy: 0.7759
#    * Corresponding 1-min(DI, 1/DI) value: 0.5738
#    * Corresponding average odds difference value: -0.2057
#    * Corresponding statistical parity difference value: -0.2612
#    * Corresponding equal opportunity difference value: -0.2228
#    * Corresponding Theil index value: 0.0921
#    
# _Random Forest Fairness Metrics_
#    * Threshold corresponding to best balanced accuracy: 0.2300
#    * Best balanced accuracy: 0.7638
#    * Corresponding 1-min(DI, 1/DI) value: 0.5141
#    * Corresponding average odds difference value: -0.1388
#    * Corresponding statistical parity difference value: -0.2190
#    * Corresponding equal opportunity difference value: -0.1135
#    * Corresponding Theil index value: 0.0936
# 
# ### Part-02: Please write one paragraph for each question.
# 1. How can we determine which metrics to use, given our data and use case? You can refer to [Course material](https://nanrahman.github.io/capstone-responsible-ai/weeks/06-Fairness-Assessments/), online research and Guidance provided by [AIF360](http://aif360.mybluemix.net/resources#)
# 2. When you have competing fairness metrics, how to pick which to prioritize?
# 3. What do you do when you encounter different definitions for similar metrics?
# 4. Based on this, which model and fairness metric appears the best to proceed with?

# ### [4.](#Table-of-Contents) Additional Model Development
# 
# 

# ### 4.1A Load data & create splits for learning/validating/testing model

# In[26]:


df_panel_19_reduced['RACE'] = (df_panel_19_reduced['RACE'] != 'Non-White').astype(int)


# In[27]:


# Use the same methods from Section 3
one_hot = pd.get_dummies(df_panel_19_reduced, columns=categorical_features, drop_first=True)
one_hot = one_hot.sample(frac=1,random_state=57)
train, val, test = BinaryLabelDataset(df=one_hot,
                                      label_names=["UTILIZATION"],
                                      protected_attribute_names=["RACE"], 
                                      privileged_protected_attributes = [0]).split([0.5, 0.8])

sens_ind = 0
sens_attr = train.protected_attribute_names[sens_ind]

unprivileged_groups = [{sens_attr: v} for v in
                       train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                     train.privileged_protected_attributes[sens_ind]]


# ### 4.1B Utilize findings from your EDA to complete any additional model development

# In[28]:


EDA_model = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='saga', max_iter=5000))
EDA_model


# In[29]:


gs = GridSearchCV(EDA_model, 
                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
                  n_jobs = -1, 
                  cv = 10)
gs.fit(train.features, train.labels.ravel())


# In[30]:


gs.score(val.features, val.labels.ravel())


# In[31]:


gs.best_estimator_.get_params()


# Actual to keep after convergence:
# 
# ['AGE', 'RACE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION_2',
#        'REGION_3', 'REGION_4', 'SEX_2', 'MARRY_3', 'MARRY_5', 'MARRY_6',
#        'MARRY_7', 'MARRY_8', 'MARRY_9', 'MARRY_10', 'FTSTU_3', 'ACTDTY_3',
#        'ACTDTY_4', 'HONRDC_4', 'RTHLTH_1', 'RTHLTH_3', 'RTHLTH_4',
#        'RTHLTH_5', 'MNHLTH_2', 'MNHLTH_4', 'MNHLTH_5', 'HIBPDX_2',
#        'CHDDX_1', 'ANGIDX_1', 'MIDX_1', 'OHRTDX_1', 'EMPHDX_1',
#        'CHBRON_1', 'CHOLDX_1', 'CHOLDX_2', 'CANCERDX_1', 'DIABDX_1',
#        'JTPAIN_2', 'ARTHDX_1', 'ARTHDX_2', 'ARTHTYPE_2', 'ARTHTYPE_3',
#        'ASTHDX_2', 'ADHDADDX_1', 'ADHDADDX_2', 'PREGNT_1', 'PREGNT_2',
#        'WLKLIM_2', 'ACTLIM_1', 'SOCLIM_2', 'COGLIM_2', 'DFHEAR42_1',
#        'DFHEAR42_2', 'DFSEE42_1', 'ADSMOK42_1', 'ADSMOK42_2', 'PHQ242_1',
#        'PHQ242_2', 'PHQ242_3', 'PHQ242_4', 'PHQ242_5', 'PHQ242_6',
#        'EMPST_2', 'EMPST_3', 'EMPST_4', 'POVCAT_2', 'POVCAT_3',
#        'POVCAT_4', 'POVCAT_5', 'INSCOV_3']

# In[32]:


keep = (gs.best_estimator_[1].coef_ != 0)[0]
to_keep = np.array(train.feature_names)[keep]
to_keep


# In[33]:


one_hot_prime = one_hot[to_keep]
one_hot_prime["UTILIZATION"] = one_hot["UTILIZATION"]
train, val, test = BinaryLabelDataset(df=one_hot_prime,
                                      label_names=["UTILIZATION"],
                                      protected_attribute_names=["RACE"], 
                                      privileged_protected_attributes = [0]).split([0.5, 0.8])


# ### 4.2. Learning a Logistic Regression (LR) classifier on original data

# In[34]:


lr_model = make_pipeline(StandardScaler(), LogisticRegression(solver='saga',max_iter=5000))
grid_lr_model = GridSearchCV(lr_model, 
                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
                  n_jobs = -1, 
                  cv = 10)
grid_lr_model.fit(train.features, train.labels.ravel())


# In[35]:


y_val_pred_prob = grid_lr_model.predict_proba(val.features)
pos_ind = np.where(grid_lr_model.classes_ == val.favorable_label)[0][0]
best_bar = -1
best_thresh = -1
for thresh in np.linspace(0.01, 0.5, 50):
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                                  val_pred, 
                                  unprivileged_groups=unprivileged_groups, 
                                  privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh
        
y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred


# In[36]:


# balanced_accuracy, avg_odds_difference, disparate_impact, statistical_parity_difference, equal_opportunity, theil
metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
lr_val_metric_arrs = metric_arrs


# In[37]:


lr_val_metric_arrs


# In[38]:


y_test_pred_prob = grid_lr_model.predict_proba(test.features)
pos_ind = np.where(grid_lr_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred
metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
lr_test_metric_arrs = metric_arrs


# In[39]:


best_thresh


# In[40]:


lr_test_metric_arrs


# ### 4.3. Learning a Random Forest (RF) classifier on original data

# We repeat this process below for the random forest classifier:

# In[41]:


rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=90, min_samples_leaf=18))
rf_model.fit(train.features, train.labels.ravel())


# In[42]:


best_bar_rf = -1
best_thresh_rf = -1
y_val_pred_prob = rf_model.predict_proba(val.features)
pos_ind = np.where(rf_model.classes_ == val.favorable_label)[0][0]
for thresh in np.linspace(0.01, 0.5, 50):   
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh


# In[43]:


y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred


# In[44]:


# balanced_accuracy, avg_odds_difference, disparate_impact, statistical_parity_difference, equal_opportunity, theil
metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
rf_val_metric_arrs = metric_arrs


# In[45]:


rf_val_metric_arrs


# In[46]:


y_test_pred_prob = rf_model.predict_proba(test.features)
pos_ind = np.where(rf_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred
metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
rf_test_metric_arrs = metric_arrs


# In[47]:


best_thresh


# In[48]:


rf_test_metric_arrs


# ### Section 4 Write Up here
# 
# **1. For both the logistic regression and random forest classifiers learned on the original data, please include the results of your fairness metrics. For _each_ metric result briefly describe (1-2 sentences) if you saw any differences from your results in Part 3, and what that might mean.**
# 
# _Logistic Regression Fairness Metrics_
#    * Threshold corresponding to best balanced accuracy: 0.16
#        * The best threshold drops from 0.19 to 0.16. This is really just due to the fact that our models are predicting upon different features. 
#    * Best balanced accuracy: 0.7710092876549082
#        * The best balanced accuracy drops by ~0.005, which is really small however still impactful given the weight of the feature we are predicting. In addition it appears that AIF360 got a lucky draw to some extent here; their validation accuracy is lower than their test accuracy by a full percentage point. Nonetheless it appears that using more features doesn't really help here, as far as logistic regression is concerned.
#    * Corresponding 1-min(DI, 1/DI) value: 0.6054505611262369
#        * Our value for this metric is higher than AIF360's indicating slightly more favorable outcomes for the underprivileged group but in the grand scheme, since there's only a 0.03 difference in these scores and both are below 0.8, this still favors the privileged group.
#    * Corresponding average odds difference value: -0.23474333072426262
#        * We do worse than AIF360 here since our value is smaller than theirs by 0.03. However again, this is such a slim difference it likely doesn't matter all that much. 
#    * Corresponding statistical parity difference value: -0.2971043867595592
#        * Both our model and AIF360's model are both just straight up unfair given this statistic, as both fall out of the ideal range from -0.1 to 0.1. However there is only a difference of 0.03, so we didn't err by too much more here, although it is still not ideal. 
#    * Corresponding equal opportunity difference value: -0.2397179340575567
#        * We do worse than AIF360 here by 0.01. Not too much of a discrepancy but still balanced towards the privileged class here. 
#    * Corresponding Theil index value: 0.09146978019069925
#        * Our Theil index is actually better than the AIF360 alternative. Not by much (0.007) but still marginally better.
#    
# _Random Forest Fairness Metrics_
#    * Threshold corresponding to best balanced accuracy: 0.16
#        * The best threshold again drops from 0.23 to 0.16.
#    * Best balanced accuracy: 0.7764437137499931
#        * We beat out AIF360 by a full percentage point. This means that we find some information in the additional features we use that they do not.
#    * Corresponding 1-min(DI, 1/DI) value: 0.5171963657575169
#        * Again we beat out AIF360 here by a slim margin. However, again, the fact that both scores are below 0.8 still means that our model is favoring the privileged group.
#    * Corresponding average odds difference value: -0.17382288217356656
#        * We do worse than AIF360 here too, but again only by 0.04. It still represents a bias towards the privileged group but shouldn't be completely unsurmountable in bias mitigation. 
#    * Corresponding statistical parity difference value: -0.25425391632288186
#        * We again do worse than AIF360 here with a differnce of 0.04. The difference doesn't matter too much though given the sheer magnitude of distance from the fair area bounded between -0.1 and 0.1. 
#    * Corresponding equal opportunity difference value: -0.15333206276602507
#        * We do worse again by 0.04, which is not a significant difference but still balanced towards the privileged group. 
#    * Corresponding Theil index value: 0.08760278911200248
#        * Our Theil index here outperforms AIF360's by 0.06, and therefore according to this metric we are only marginally more fair.
#     
# **2. Based on this, would you make any recommendations during model development? Does it change which model and fairness metric would be the best to proceed with?** (Please write at least one paragraph)
# 
# We found that a good portion of the features contained in the dataset contained information relevant to the UTILIZATION variable, specifically 88/103 of the one-hot encoded variables we created. As such it makes sense to use a consistent EDA pipeline to discover which features make the most sense to use (in our case we use a logistic regression with l2 loss to discover which features actually help, and which are redundant). Ultimately this was the only change we made on top of some additional hyperparameterizing, and so there likely isn't a major change that should be made to the fairness metric we would optimize towards (balanced accuracy still makes the most sense; a model should be both accurate and fair). That being said, after our further analysis we found that the random forest had not only more consistent performance but also better performance on both the validation and test sets, so this might be the model of choice moving forwards.

# # Start of Replication Part 03a -  Bias Mitigation Techniques
# 
# ## There are **two** components to `Replication Project Part #03`
# 
# ### Part 1. Run the full tutorial example. Within AIF360's Repository it can be found under: `AIF360`/`Examples`/tutorial_medical_expenditure.ipynb
# 
# #### BEFORE YOU BEGIN MAKE SURE THAT:
# * A reminder, you will need to fork [AIF360's repository](https://github.com/Trusted-AI/AIF360) into your own GitHub and access the notebook locally or via your method of choice
# * AIF360's Repository can be found under: `AIF360`/`Examples`/tutorial_medical_expenditure.ipynb
# * Ensure you have your `aif360` environment turned and activated using a miniconda prompt
# * Use Jupyter Labs
# * Refer to [Week 03](https://nanrahman.github.io/capstone-responsible-ai/weeks/03-Replication-Part-00/) content on the course Website to access the `Quickstart Guide`
# 
# #### FOR THE DATA
# * Downloade the `h181.csv` and `h192.csv` data files uploaded [here](https://www.kaggle.com/datasets/nanrahman/mepsdata)
# * Place the `h181.csv` and `h192.csv` ino `../aif360/data/raw/meps/` of your forked AIF360 repository
# 
# ### Part 2. Training models WITH de-biasing, trying out another type of de-biasing method
# 
# *Below is a list of additional notebooks that demonstrate the use of AIF360*
# 
# * NEW: [sklearn/demo_new_features.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/sklearn/demo_new_features.ipynb): highlights the features of the new scikit-learn-compatible API
# * [demo_optim_data_preproc.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb): demonstrates a generalization of the credit scoring tutorial that shows the full machine learning workflow for the optimized data pre-processing algorithm for bias mitigation on several datasets
# * [demo_adversarial_debiasing.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_adversarial_debiasing.ipynb): demonstrates the use of the adversarial debiasing in-processing algorithm to learn a fair classifier
# * [demo_calibrated_eqodds_postprocessing.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_calibrated_eqodds_postprocessing.ipynb): demonstrates the use of an odds-equalizing post-processing algorithm for bias mitigiation
# * [demo_disparate_impact_remover.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_disparate_impact_remover.ipynb): demonstrates the use of a disparate impact remover pre-processing algorithm for bias mitigiation
# * [demo_lfr.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_lfr.ipynb): demonstrates the use of the learning fair representations algorithm for bias mitigation
# * [demo_lime.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_lime.ipynb): demonstrates how LIME - Local Interpretable Model-Agnostic Explanations - can be used with models learned with the AIF 360 toolkit to generate explanations for model predictions
# * [demo_reject_option_classification.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_reject_option_classification.ipynb): demonstrates the use of the Reject Option Classification (ROC) post-processing algorithm for bias mitigation
# * [demo_reweighing_preproc.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_reweighing_preproc.ipynb): demonstrates the use of a reweighing pre-processing algorithm for bias mitigation
# 

# ---

# ## [5.](#Table-of-Contents) Bias Mitigation

# ### [5A.](#Table-of-Contents) Bias mitigation using pre-processing technique, Reweighing - AIF360 Example
# 

# In[49]:


RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
RW_train = RW.fit_transform(train)


# ### 5B. LR trained, validated, and tested using reweighted data

# In[50]:


lr_model = make_pipeline(StandardScaler(), LogisticRegression(solver='saga',max_iter=5000))
RW_grid_lr_model = GridSearchCV(lr_model, 
                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
                  n_jobs = -1, 
                  cv = 10)
RW_grid_lr_model.fit(RW_train.features, RW_train.labels.ravel())


# In[51]:


y_val_pred_prob = RW_grid_lr_model.predict_proba(val.features)
pos_ind = np.where(RW_grid_lr_model.classes_ == val.favorable_label)[0][0]
best_bar = -1
best_thresh = -1
for thresh in np.linspace(0.01, 0.5, 50):
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                                  val_pred, 
                                  unprivileged_groups=unprivileged_groups, 
                                  privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh
        
y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred


# In[52]:


metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
RW_lr_val_metric_arrs = metric_arrs


# In[53]:


lr_val_metric_arrs


# In[54]:


RW_lr_val_metric_arrs


# In[55]:


y_test_pred_prob = RW_grid_lr_model.predict_proba(test.features)
pos_ind = np.where(RW_grid_lr_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred
metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
RW_lr_test_metric_arrs = metric_arrs


# In[56]:


best_thresh


# In[57]:


lr_test_metric_arrs


# In[58]:


RW_lr_test_metric_arrs


# ### RF trained, validated, and tested using reweighted data

# In[59]:


RW_rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=90, min_samples_leaf=18))
RW_rf_model.fit(RW_train.features, RW_train.labels.ravel())


# In[60]:


best_bar_rf = -1
best_thresh_rf = -1
y_val_pred_prob = RW_rf_model.predict_proba(val.features)
pos_ind = np.where(RW_rf_model.classes_ == val.favorable_label)[0][0]
for thresh in np.linspace(0.01, 0.5, 50):   
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh


# In[61]:


y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred


# In[62]:


# balanced_accuracy, avg_odds_difference, disparate_impact, statistical_parity_difference, equal_opportunity, theil
metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
RW_rf_val_metric_arrs = metric_arrs


# In[63]:


rf_val_metric_arrs


# In[64]:


RW_rf_val_metric_arrs


# In[65]:


y_test_pred_prob = RW_rf_model.predict_proba(test.features)
pos_ind = np.where(RW_rf_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred
metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
RW_rf_test_metric_arrs = metric_arrs


# In[66]:


rf_test_metric_arrs


# In[67]:


RW_rf_test_metric_arrs


# ### [5B.](#Table-of-Contents) Prejudice Remover (in-processing bias mitigation) -  AIF360 Example
# 

# In[70]:


pr_model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
scaler = StandardScaler()

new_train = train.copy()
new_train.features = scaler.fit_transform(new_train.features)
pr_model = model.fit(new_train)


# In[79]:


new_val = val.copy()
new_val.features = scaler.transform(new_val.features)

best_bar_pr = -1
best_thresh_pr = -1
y_val_pred_prob = model.predict(new_val).scores
pos_ind = 0
for thresh in np.linspace(0.01, 0.5, 50):   
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = new_val.labels.ravel()
    new_val_pred = new_val.copy()
    new_val_pred.labels = y_val_pred
    metric = ClassificationMetric(new_val, 
                              new_val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar_pr:
        best_bar_pr = curr_bar
        best_thresh_pr = thresh
        
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
pr_val_metric_arrs = metric_arrs
pr_val_metric_arrs


# In[80]:


best_thresh_pr


# In[86]:


new_test = test.copy()
new_test.features = scaler.transform(new_test.features)

y_test_pred_prob = pr_model.predict(new_test).scores
pos_ind = 0
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh_pr).astype(np.float64)
y_true = new_test.labels.ravel()
new_test_pred = new_test.copy()
new_test_pred.labels = y_test_pred
metric = ClassificationMetric(new_test, 
                              new_test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)

metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
pr_test_metric_arrs = metric_arrs


# In[87]:


pr_test_metric_arrs


# ### [5C.](#Table-of-Contents) Bias mitigation using a technique of your own
# 

# In[88]:


from aif360.algorithms.preprocessing import LFR


# In[89]:


fair_rep = LFR(unprivileged_groups, privileged_groups)
train = fair_rep.fit_transform(train)


# ### LR model training, validation and testing after fair representations

# In[90]:


lr_model = make_pipeline(StandardScaler(), LogisticRegression(solver='saga',max_iter=5000))
grid_lr_model = GridSearchCV(lr_model, 
                  {'logisticregression__C': list(np.logspace(-5, 5, 10))}, 
                  n_jobs = -1, 
                  cv = 10)
grid_lr_model.fit(train.features, train.labels.ravel())


# In[96]:


y_val_pred_prob = grid_lr_model.predict_proba(val.features)
pos_ind = np.where(grid_lr_model.classes_ == val.favorable_label)[0][0]
best_bar = -1
best_thresh = -1
for thresh in np.linspace(0.01, 0.5, 50):
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                                  val_pred, 
                                  unprivileged_groups=unprivileged_groups, 
                                  privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh
        
y_val_pred = (y_val_pred_prob[:, pos_ind] <= best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred

metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
lr_val_metric_arrs = metric_arrs
lr_val_metric_arrs


# In[97]:


y_test_pred_prob = grid_lr_model.predict_proba(test.features)
pos_ind = np.where(grid_lr_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] <= best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred

metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
lr_test_metric_arrs = metric_arrs


# In[98]:


lr_test_metric_arrs


# ### RF model training, validation, and testing after learning fair representations

# In[99]:


rf_model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=90, min_samples_leaf=18))
rf_model.fit(train.features, train.labels.ravel())


# In[100]:


best_bar_rf = -1
best_thresh_rf = -1
y_val_pred_prob = rf_model.predict_proba(val.features)
pos_ind = np.where(rf_model.classes_ == val.favorable_label)[0][0]
for thresh in np.linspace(0.01, 0.5, 50):   
    y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)
    y_true = val.labels.ravel()
    val_pred = val.copy()
    val_pred.labels = y_val_pred
    metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
    curr_bar = (metric.true_positive_rate() + metric.true_negative_rate()) / 2
    if curr_bar > best_bar:
        best_bar = curr_bar
        best_thresh = thresh
y_val_pred = (y_val_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = val.labels.ravel()
val_pred = val.copy()
val_pred.labels = y_val_pred
metric = ClassificationMetric(val, 
                              val_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
rf_val_metric_arrs = metric_arrs
rf_val_metric_arrs


# In[101]:


y_test_pred_prob = rf_model.predict_proba(test.features)
pos_ind = np.where(rf_model.classes_ == test.favorable_label)[0][0]
y_test_pred = (y_test_pred_prob[:, pos_ind] > best_thresh).astype(np.float64)
y_true = test.labels.ravel()
test_pred = test.copy()
test_pred.labels = y_test_pred
metric = ClassificationMetric(test, 
                              test_pred, 
                              unprivileged_groups=unprivileged_groups, 
                              privileged_groups=privileged_groups)
metric_arrs = {}
metric_arrs['bal_acc']= (metric.true_positive_rate() + metric.true_negative_rate()) / 2
metric_arrs['avg_odds_diff'] = metric.average_odds_difference()
metric_arrs['disp_imp'] = metric.disparate_impact()
metric_arrs['stat_par_diff'] = metric.statistical_parity_difference()
metric_arrs['eq_opp_diff'] = metric.equal_opportunity_difference()
metric_arrs['theil_ind'] = metric.theil_index()
rf_test_metric_arrs = metric_arrs


# In[102]:


rf_test_metric_arrs


# ---

# # Replication Part 03b -  Write Up Instructions
# * Look for this in Week 07

# 
