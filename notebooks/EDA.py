import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\hp\Documents\ML\bank\bank-additional-full.csv", sep=';')
df.head()
df.info()
df.describe(include='all')
df.isnull().sum()


#Observation de la variable cible
sns.countplot(data=df, x='y')
plt.title("Distribution de la variable cible (yes/no)")
plt.show()

df['y'].value_counts(normalize=True)


#Analyse des variables démographiques
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['age'], kde=True, ax=ax[0])
ax[0].set_title("Distribution de l'âge")

sns.boxplot(data=df, x='y', y='age', ax=ax[1])
ax[1].set_title("Âge vs Souscription")

plt.show()

#Pour les variables catégorielles :
categorical_cols = ['job','marital','education']

for col in categorical_cols:
    plt.figure(figsize=(12,4))
    sns.countplot(data=df, x=col, hue='y')
    plt.title(f"{col} par souscription")
    plt.xticks(rotation=45)
    plt.show()

 #Analyse des variables de campagne
plt.figure(figsize=(8,4))
sns.histplot(df['duration'], kde=True)
plt.title("Durée de l'appel")
plt.show()

sns.boxplot(data=df, x='y', y='duration')
plt.title("Durée de l'appel vs Souscription")
plt.show()


#Analyse des variables économiques
econ_vars = ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m']

sns.pairplot(df[econ_vars + ['y']], hue='y')
plt.show()

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

#Correlation des variables numériques à la variable cible
plt.figure(figsize=(12,8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Corrélation entre variables numériques")
plt.show()

#Analyse bivariée cible vs feature
pd.crosstab(df['job'], df['y'], normalize='index').plot(kind='bar', figsize=(12,5))
plt.title("Taux de souscription par métier")
plt.show()


