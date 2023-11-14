# EX 10- Mini-Project
## DATE:
## Rainfall Analysis of India
### DescrIption : 
As is widely recognized, rainfall data is necessary for the mathematical modelling of extreme hydrological events, such as droughts or floods, as well as for evaluating surface and subsurface water resources and their quality. The phase, quantity, and elevation of generic hydrometeors in the atmosphere can be estimated by ground-based radars. Satellites can provide images with visible and infrared radiation, and they can also serve as platforms for radiometers to derive the quantity and phase of hydrometeors. Radars and satellites provide spatial information on precipitation at wide scales, avoiding many problems connected to local ground measurements, including those for the areal inhomogeneity of a network. However, direct rainfall observations at point scale can be obtained only by rain gauges installed at the soil surface.

### KEY FEATURES : 
Three main characteristics of rainfall are its amount, frequency and intensity, the values of which vary from place to place, day to day, month to month and also year to year. Precise knowledge of these three main characteristics is essential for planning its full utilization.

## CODE:
```
DEVELOPED BY: HARINI V
REGISTER NO: 212222230044
```
```
PYTHON
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import train_test_split

#read the whole file 
df.iloc[:]
df.info()
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
df.columns
row3 = df.iloc[3]
print(row3)
print("\n")
row115 = df.iloc[115]
print (row115)
description = df.describe()
print(description)
#plot the actual rain 
dg = df[['jun','jul','aug','sep']]
dg.plot(figsize=(20, 5));
#plot shows the total rain 
dh = df[['total']]
dh.plot(figsize=(10, 10));
#dy represent dataframe for 10 last rows which is rainfall from year 2007 to 2016 
dy = df.tail(10)
print(dy)
#plot diagram shows only "TOTAL" rain from dy dataframe (rainfall from year 2007 to 2016)
de = dy[['total']]
de.plot(figsize=(10, 10));
#rainfall from 2007 to 2016
#plot actual rain and departure percentage 
a_d = dy[['jun', 'jul', 'aug', 'sep', 'jun_p', 'jul_p', 'aug_p', 'sep_p']]
a_d.plot(figsize=(20, 10));
#rainfall from 2007 to 2016
#only plot actual rain 
actual = dy[['jun', 'jul', 'aug', 'sep']]
actual
actual = dy[['jun', 'jul', 'aug', 'sep']]
# Calculate the rainfall ranges for each month
rainfall_ranges = actual.max() - actual.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = actual[month].min()
    max_rainfall = actual[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} mm,", end=' ')
    print(f"Min = {min_rainfall} mm, Max = {max_rainfall} mm")
actual.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Actual Rainfall (mm) vs Year')
plt.xlabel('Year')
plt.ylabel('Actual Rainfall (mm)')
plt.show()
#rainfall from 2007 to 2016
#plot departure percentage 
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
de
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
# Calculate the rainfall ranges for each month
rainfall_ranges = de.max() - de.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = de[month].min()
    max_rainfall = de[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} %,", end=' ')
    print(f"Min = {min_rainfall} %, Max = {max_rainfall} %")
print("\n")
de.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Departure Percentage vs Year')
plt.xlabel('Month')
plt.ylabel('Departure Percentage (%)')
plt.show()
#From 1901 to 2016
actual_df = df[['jun', 'jul', 'aug', 'sep']]
actual_df
# Calculate the average rainfall for each month
av_df = actual_df.mean()
av_df
#Stacked area chart
#Calculate the cumulative rainfall for each month or season
cumulative_rainfall = df[['jun', 'jul', 'aug', 'sep']].cumsum(axis=1)
# Plot the stacked area chart
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, cumulative_rainfall.values.T, labels=['Jun', 'Jul', 'Aug', 'Sep'])
plt.title('Cumulative Rainfall Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall (mm)')
plt.legend(loc='upper left')
plt.show()
#Scatter plot
# Create a scatter plot with trendline
plt.figure(figsize=(10, 6))
sns.regplot(x=actual_df.index, y='jun', data=actual_df, scatter=True, label='June')
sns.regplot(x=actual_df.index, y='jul', data=actual_df, scatter=True, label='July')
sns.regplot(x=actual_df.index, y='aug', data=actual_df, scatter=True, label='August')
sns.regplot(x=actual_df.index, y='sep', data=actual_df, scatter=True, label='September')
plt.title('Rainfall Variation over Time')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
actual_df
# Plotting the predicted values versus the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
```
OUTPUT:
![image](https://github.com/harini1006/Mini-Project/assets/113497405/d077a5fc-845d-4d8d-9e7b-1c2f1ce70963)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/ff5effe4-1675-4d49-8c92-d786854faee8)






```
print(f'Number of Rows: {df.shape[0]}') : to print the number of rows
print(f'Number of Columns: {df.shape[1]}') : to print number of columns
```
![image](https://github.com/harini1006/Mini-Project/assets/113497405/1e5e5c78-099e-4902-abbf-e19a1dafa01b)
```
df.columns is to retrieve the column labels or column names of a DataFrame to shows data at row 3 (year 1904) and row 115 (year 2016)
```
![image](https://github.com/harini1006/Mini-Project/assets/113497405/1828248a-dfac-4f0b-8619-fb19761af4be)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/d2f9f71c-529c-4970-8344-89fcc8c4d143)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/a0ee0ac0-544d-4f68-9628-4b5dce43e3cc)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/cb5b7cdd-2848-4305-80ef-40c83464319c)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/658daacf-5519-40af-abab-49f6b4ec7180)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/3b4d48c4-e5ce-4238-87e3-e153c640157a)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/5b4a6a05-050b-45c0-b540-63ecc2de5c6c)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/3ec2ffd0-02fc-4bb0-bde7-713be50d9aaf)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/238b9811-a0f4-42db-9279-a349a0980f70)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/df7f9b45-3aec-49e1-970a-5fab2fe9d2ea)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/553f80b1-6267-4ddb-9bde-b626ba27e7ba)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/600fa274-d5db-45fe-9b89-4414054eff86)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/0d34d2b3-0547-4877-9c5e-aeb258c15947)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/b31a022b-80f2-489c-994c-ccd1c8c5a00e)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/19d9cc33-47bf-4d10-8ed0-6907874e45a9)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/3a218b83-cdcb-4e32-a6ce-34a5528e8fc4)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/53262390-d1f3-4f4e-b89d-986e26e9d753)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/b9f054d1-f8fa-43fd-bd90-c08137cff6bb)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/720ccfee-3fcf-45a5-a113-39a4a153906c)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/50834405-25e4-4bb5-a30b-f02c7b8b8445)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/ada6f9de-e1b5-4d53-8656-1e7b97ce2f4d)



Based on the cumulative rainfall data for the months of June, July, August, and September from 1901 to 2016, The cumulative rainfall values represent the total amount of rainfall received from June to September for each year. The data shows variations in cumulative rainfall over time, indicating fluctuations in precipitation patterns.

![image](https://github.com/harini1006/Mini-Project/assets/113497405/207c95ad-e0ac-43e1-934f-efcd8700bd85)


![image](https://github.com/harini1006/Mini-Project/assets/113497405/f38e6981-9b4e-401e-a943-9dc5e404ceb9)



## RESULT
Thus the rainfall analysis has been executed successfully.
