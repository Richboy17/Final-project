#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import the dataset
redwine = pd.read_csv("/content/winequality-red.csv", sep=';')

redwine

redwine.describe()

sns.countplot(x = redwine['quality']).set(title = 'Quality Distribution', 
                                       xlabel = 'Quality', 
                                       ylabel = 'Count');
 
winecopy = redwine.copy()
 
winecopy
 
winecopy.duplicated().value_counts()

winecopy_clean = winecopy.drop_duplicates(inplace=False)
winecopy_clean

winecopy_clean["quality"] = winecopy_clean["quality"].astype('category', copy=False)
winecopy_clean

sns.lineplot(data = winecopy_clean, x = 'quality', y = 'density')

plt.figure(figsize=(5,5))
sns.scatterplot(x = v1, y = winecopy_clean['citric acid'])

sns.lineplot(data = winecopy_clean, x = v1, y = winecopy_clean['fixed acidity'])

plt.figure(figsize=(20,5))
sns.barplot(data = winecopy_clean, x = v1, y = winecopy_clean['volatile acidity'])

sns.lineplot(data = winecopy_clean, x = 'free sulfur dioxide', y = 'total sulfur dioxide')

plt.figure(figsize=(15,10))
sns.heatmap(redwine.corr(),linewidth=0.5,annot=True, center=0, cmap='GnBu')

winecopydrop = winecopy_clean.drop('total sulfur dioxide', axis=1)

winecopydrop1 = winecopydrop.drop('pH', axis=1)

winecopydrop2 = winecopydrop1.drop('citric acid', axis=1)

x = winecopydrop2.iloc[:,:8]
y = winecopydrop2['quality']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=100) 

mlr= LinearRegression()  
mlr.fit(x_train, y_train) 

#Printing the model coefficients
print(mlr.intercept_)
# pair the feature names with the coefficients
list(zip(x, mlr.coef_))

y_pred_mlr= mlr.predict(x_test)  
x_pred_mlr= mlr.predict(x_train)  

mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff

print('R squared value of the model: {:.2f}'.format(mlr.score(x,y)*100))

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)