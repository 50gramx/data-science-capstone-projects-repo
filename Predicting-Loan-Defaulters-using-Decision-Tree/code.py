# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)
X = data.drop(['customer.id', 'paid.back.loan'], axis=1)
y = data['paid.back.loan'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
fully_paid.plot(kind='bar')


# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
# Removing the last character from the values in column
X_train['int.rate'] = X_train['int.rate'].map(lambda x: str(x)[:-1])

#Dividing the column values by 100
X_train['int.rate'] = X_train['int.rate'].astype(float)/100

# Removing the last character from the values in column
X_test['int.rate'] = X_test['int.rate'].map(lambda x: str(x)[:-1])

#Dividing the column values by 100
X_test['int.rate'] = X_test['int.rate'].astype(float)/100

# Subset of only numerical columns of X_train
num_df = X_train.select_dtypes(include=['number']).copy()
cat_df = X_train.select_dtypes(include=['object']).copy()




# Code ends here


# --------------
#Importing header files
import seaborn as sns

# Setting the figure size
plt.figure(figsize=(20,20))

# Storing the columns of 'num_df'
cols = list(num_df.columns)

# Creating subplots
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10,20))

# Looping across rows
for i in range(9):
    
    # Plotting boxplot
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])

    # Avoiding subplots overlapping
    fig.tight_layout()


# --------------
# Creating a list of column names of cat_df
cols = list(cat_df.columns)

# Setting the figure size
plt.figure(figsize=(20,20))

# Creating the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

# Looping through rows
for i in range(2):

    # Looping through columns
    for j in range(2):

            # Plotting count plot
            sns.countplot(x=X_train[cols[i*2 + j]], hue=y_train, ax=axes[i,j])

            # Avoiding subplots overlapping
            fig.tight_layout()



# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Fill the null values with 'NA'
X_train.fillna('NA')
X_test.fillna('NA')

# Iterate for categorical columns
for col in cat_df.columns:

    # Fill null values with 'NA'
    X_train[col].fillna('NA', inplace=True)

    # Initalising a label encoder object
    le = LabelEncoder()

    # Fitting and transforming the column in X_train with 'le'
    X_train[col] = le.fit_transform(X_train[col])

    # Filling null values with 'NA'
    X_test[col] = le.transform(X_test[col])

# Replacing the values of y_train
y_train.replace({'No': 0, 'Yes': 1}, inplace=True)

# Replacing the values of y_test
y_test.replace({'No': 0, 'Yes': 1}, inplace=True)

# Initialisin 'Decision Tree' Model
model = DecisionTreeClassifier(random_state=0)

# Training the 'Decision Tree' model
model.fit(X_train, y_train)

# Finding the accuracy of 'Decision Tree' Model
acc = model.score(X_test, y_test)

# Printing the accuracy
print(acc)





# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

model_2 = DecisionTreeClassifier(random_state=0)
p_tree = GridSearchCV(estimator=model_2, param_grid = parameter_grid,cv=5)
p_tree.fit(X_train, y_train)

acc_2 = p_tree.score(X_test, y_test)
print(acc_2)



# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here

dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])

graph_big = pydotplus.graph_from_dot_data(dot_data)

Image(graph_big.create_png())

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


