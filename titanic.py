# Here I performed some high level cleaning in the famous Titanic data that is available on Kaggle and many other websites.
# This processing was showed on my YouTube channel "O Cientista de Dados".

##############
### PART I ###
##############

## Reading and Cleaning the dataset (data munging) ##

# Importing Packages/Libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
# %matplotlib inline

# Open CSV file:
Jack = pd.read_csv('titanic.csv', header=0) # in general, in name the file as "df"

#with open('titanic.csv') as f:
    #df = pd.read_csv(f, dtype={'Age': np.float64})

# Quick overview:
Jack.head() # set specific number of 10 - NaN (Not a Number)
Jack.tail()

Jack.shape # number rows and number of columns

# See "all data":
print(Jack)

# Check columns' names:
Jack.columns

# Check data types:
Jack.info()
Jack.dtypes

# Basic statistics:
Jack.describe() # be aware of potential null values and outliers

Jack['fare'].describe() # basic statistics for a particular column

# Check for NULL values:
Jack.isnull()

Jack.isnull().sum() # in Excel use "COUNTBLANK" + conditional formating

sns.heatmap(Jack.isnull(), cmap="YlGnBu", cbar=False) # viz null values

print(1188/1309*100)

# Managing Null Values:
Jack['embarked'] = Jack.embarked.fillna('Q') # fill blank w/ specific value

Jack['age'] = Jack.age.fillna(Jack.age.median()) # fill blanks w/ median

Jack.dropna(subset=['fare'], inplace=True) # drop/delete the row w/ missing value

Jack.isnull().sum()

sns.heatmap(Jack.isnull(), cmap="Paired", cbar=False)

# Create new data frame w/ columns without missing values"
Rose = Jack[['pclass','survived','sex', 'age', 'sibsp', 'parch',
       'fare','embarked']]

Rose.columns

sns.heatmap(Rose.isnull(), cmap="Set3", cbar=False)
Rose.shape
Rose.head()

# Create new CSV File
df.to_csv('Rose.csv')

#print(Rose['survived'])


###############
### PART II ###
###############

## Analysis ##

# Check Correlation - Heatmap:
Rose.corr()


sns.heatmap(Rose.corr(), cmap="YlGnBu", linewidths=.5, annot=True)
plt.title('Heatmap for Titanic Data', fontsize=15)
plt.savefig('tt_heatmap.jpeg')
#plt.show()


# Boxplot:
Rose_box = Rose.boxplot(column=['age'], grid=False)
plt.title("Boxplot for Age")


# Stacked barplot Survival by Gender:
## Sex survival"
sex_survival = Rose[Rose['survived']==1]['sex'].value_counts()
sex_dead = Rose[Rose['survived']==0]['sex'].value_counts()
df = pd.DataFrame([sex_survival,sex_dead])
df.index = ['Sobreviveram','Morreram']
df.plot(kind='bar',stacked=True, figsize=(8,6), color=['purple','green'])

## Embarked by location (C = Cherbourg, Q = Queenstown, S = Southampton):
embark_survived = Rose[Rose['survived']==1]['embarked'].value_counts()
embark_dead = Rose[Rose['survived']==0]['embarked'].value_counts()
df = pd.DataFrame([embark_survived,embark_dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(8,6), 
        color=['black', 'silver', 'white'] )



# Nested barplot:
## Survival probability plot 1:
sp = sns.catplot(x="pclass", y="survived", hue="sex", data=Rose,
                height=6, kind="bar", palette="Set3")
sp.despine(left=True)
sp.set_ylabels("survival probability")

## Survival probability plot 2 (C = Cherbourg, Q = Queenstown, S = Southampton):
sp = sns.catplot(x="embarked", y="survived", hue="sex", data=Rose,
                height=6, kind="bar", palette="Set3")
sp.despine(left=True)
sp.set_ylabels("survival probability")


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


bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
Rose['age_binned'] = pd.cut(Rose.age, bins, right=True, include_lowest=False)
# child variable
Rose['Child'] = (Rose.age < 16).astype(int)
surv = Rose[Rose.survived == 1]


fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2,figsize=(12,14))

sns.set_style('dark')
order = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]',
         '(60, 70]', '(70, 80]']

sns.countplot(x='age_binned', color='white', ax=ax1, data=Rose, order=order)
sns.countplot(x='age_binned', ax=ax1, data=Rose, order=order)
plt.xlabel('Age Group')
plt.ylabel('Total / Survived')
ax1.set_title('Age')
ax1.set_xlabel('Age Group')
ax1.set_ylabel('Number Survived')


sns.set_style('dark')
sns.countplot(x='Sex', color='white', ax=ax2, data=df, order=['male','female'])
sns.countplot(x='Sex', ax=ax2, data=surv,  order=['male','female'])
plt.ylabel('Survived')
ax2.set_xlabel('')
ax2.set_title('Gender')
ax2.set_ylabel('')

sns.countplot(x='Pclass', color='white', ax=ax3, data=df,  order=[1, 2, 3])
sns.countplot(x='Pclass', ax=ax3, data=surv,  order=[1, 2, 3])
ax3.set_title('Passenger Class')
ax3.set_ylabel('Number Survived')
ax3.set_xlabel('Class')


sns.countplot(x='Child', color='white', ax=ax4, data=df, order=[1,0])
sns.countplot(x='Child', ax=ax4, data=surv, order=[1,0])
loc, labels = plt.xticks()
plt.xticks(loc,['Child (<16 yrs)','Not Child'])
plt.ylabel('Survived')
ax4.set_title('Children')
ax4.set_ylabel('')

# data source:
# https://www.kaggle.com/c/titanic/data
