# Here I performed some data munging and analytics using the famous Titanic dataset:
# You can see me doing that on my YouTube channel (in Portuguese) called "O Cientista de Dados".

###############
### PART I ####
###############

## Reading and Cleaning/Transformation of the dataset (data munging) ##

# Importing Packages/Libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression # scikit-learn for ML
# %matplotlib inline

##########################################
# Open CSV file ( from Kaggle - "cagow") #
##########################################

Jack = pd.read_csv('titanic.csv', header=0) # in general, in name the file as "df"


# Quick overview:
Jack.head(10) # set specific number of 10 - NaN (Not a Number)
Jack.tail(22)


# See "all data":
print(Jack)
Jack

# Pulling info by index:
Jack.iloc[2]


# number rows and number of columns
Jack.shape


# Check columns' names:
Jack.columns


# Check data types:
Jack.info()
Jack.dtypes


# Basic statistics:
Jack.describe() # be aware of potential null values and outliers
Jack['fare'].describe() # basic statistics for a particular column


############################
# Check for missing values #
############################

Jack.isnull()

Jack.isnull().sum() # in Excel use "COUNTBLANK" + conditional formating

# Heatmap for displaying null values:
sns.heatmap(Jack.isnull(), cmap="YlGnBu", cbar=False) # viz null values

print(1188/1309*100)


# Managing Missing Values (if you do ANY imputation techniques, you MUST include an additional field specifying that field that was imputed):

# Why is the data missing?


####################################################
# Duplicating columns where values will be imputed #
####################################################


# Fill 'age' column/rows w median:
Jack['age'] = Jack.age.fillna(Jack.age.median()) # fill blanks w median
Jack.info()



# Clone colum 'embarked':
Jack['embarked_2'] = Jack['embarked']
Jack.info()

# Create binary column for indication:
Jack_dummy = pd.get_dummies(Jack, dummy_na=True, columns=['embarked'])

Jack_dummy.info()

# Create imputed column:
Jack_dummy['embarked_imputed'] = Jack['embarked_2']
Jack_dummy.info()

# Fill column/rows w categorical data (Q):
Jack_dummy['embarked_imputed'] = Jack.embarked.fillna('Q') # fill blank w/ specific value

Jack_dummy.info()

# Rename Columns:
Jack_dummy.rename(columns = {'embarked_2':'embarked'}, inplace = True)
Jack_dummy.rename(columns = {'embarked_nan':'embarked_imputed_IND'}, inplace = True)

# Delete Columns:
Jack_dummy.drop(['embarked_C', 'embarked_Q','embarked_S'], axis=1, inplace=True)

Jack_dummy.info()

# Export data frame to csv file:
Jack_dummy.to_csv('Jack_dummy.csv', index=False)



# Dropping missing value on column 'fare':
Jack_dummy.dropna(subset=['fare'], inplace=True) # drop/delete the row w/ missing value

Jack_dummy.isnull().sum()

sns.heatmap(Jack_dummy.isnull(), cmap="Paired", cbar=False)




############################################################
# Create new data frame w/ columns without missing values" #
############################################################

Rose = Jack_dummy[['pclass','survived','sex', 'age', 'sibsp', 'parch',
       'fare', 'embarked', 'embarked_imputed', 'embarked_imputed_IND']]

Rose.columns

sns.heatmap(Rose.isnull(), cmap="Set3", cbar=False)
Rose.shape
Rose.head()
Rose.columns



#######################
# Create new CSV File #
#######################
Rose.to_csv('Rose.csv', index=False)




#####################
#####################
# Part II: Analysis #
#####################
#####################


# Using Spyder and Jupyter
# Check for Missing values:
sns.heatmap(Rose.isnull(), cmap="Set3", cbar=False)

# Count of Survivals:
ax = sns.countplot(x="survived", data=Rose)
plt.title("Count of Survivals")

# Check Correlation:
Rose.corr()

# Heatmap for Correlation:
sns.heatmap(Rose.corr(), cmap="YlGnBu", linewidths=.5, annot=False)
plt.title('Heatmap for Titanic Data', fontsize=15)
plt.savefig('tt_heatmap.jpeg')
#plt.show()


# Boxplot:
Rose_box = Rose.boxplot(column=['age'], grid=False)
plt.title("Boxplot for Age")


ax = sns.boxplot(x="embarked", y="age", data=Rose)


# Stacked barplot Survival by Gender:
## Sex survival"
sex_survival = Rose[Rose['survived']==1]['sex'].value_counts()
sex_dead = Rose[Rose['survived']==0]['sex'].value_counts()
df = pd.DataFrame([sex_survival,sex_dead])
df.index = ['Sobreviveram','Morreram']
df.plot(kind='bar',stacked=True, figsize=(8,6), color=['purple','green'])


## Embarked by location (C = Cherbourg, Q = Queenstown, S = Southampton):
embark_survived = Rose[Rose['survived']==1]['embarked'].value_counts()
embark_dead = Rose[Rose['survived']==0]['embarked_imputed'].value_counts()
df = pd.DataFrame([embark_survived,embark_dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(8,6), 
        color=['black', 'silver', 'yellow'] )


# Nested barplot:
## Survival probability plot 1:
sp = sns.catplot(x="pclass", y="survived", hue="sex", data=Rose,
                height=6, kind="bar", palette="Set3")
sp.despine(left=True)
sp.set_ylabels("survival probability")


## Survival probability plot 2 (C = Cherbourg, Q = Queenstown, S = Southampton):
sp = sns.catplot(x="embarked_imputed", y="survived", hue="sex", data=Rose,
                height=6, kind="bar", palette="Set3")
sp.despine(left=True)
sp.set_ylabels("survival probability")


## Survival probability by Gender, Embarked and Pclass:
FacetGrid = sns.FacetGrid(Rose, row='embarked_imputed', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'pclass', 'survived', 'sex', palette=None,
              order=None, hue_order=None )
FacetGrid.add_legend()


## Survival probability for sibsp and parch:
# Loop for count of alones and not alones:
data = [Rose]
for dataset in data:
    dataset['relatives'] = dataset['sibsp'] + dataset['parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
Rose['not_alone'].value_counts()

# Probability plot for alones and not alones:
sns.set_palette("Set1")
axes = sns.factorplot('relatives','survived', 
                      data=Rose, aspect = 2.5, )
plt.title("Probability of Survival Based on Company")


Rose.to_csv('Rose.csv', index=False)


# Facetting histograms:
g = sns.FacetGrid(Rose, row="sex", col="survived", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "fare", color="steelblue", bins=bins)


# Hexbin Joint distributions
with sns.axes_style('white'):
    sns.jointplot("age", "fare", data=Rose, kind='hex')
plt.savefig('tt_hexbin.jpeg')


# Scatter plot:
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.scatterplot(x="age", y="fare",
                hue="survived",
                palette="husl",
                sizes=(1, 8), linewidth=0,
                data=Rose, ax=ax)


sns.relplot(x="age", y="fare", hue="survived", sizes=(200, 400), alpha=.8, 
            palette="muted", height=6, data=Rose)



# Pair plot:
sns.pairplot(Rose, hue="survived")



# https://www.kaggle.com/c/titanic/data
