import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
list_of_null_values=['null', 'na', 'n/a', 'N/A', 'missing', '_', '', '-', 0, 'none', 'nan',
    'Not Applicable', 'Unknown', 'Unspecified', '?', '--', 'null value', 'no data']
original_data_set=pd.read_csv("C:/Users/PMLS/Downloads/student_score/student_scores.csv",na_values=list_of_null_values)
print(original_data_set)
# First we will check the null values in columns
print(original_data_set.isnull().sum())
# No null values present in it 

copied_data_set=original_data_set.copy()

# checking outliers in data by plotting box plot 


# for i in copied_data_set:
#  sns.boxplot(copied_data_set[i])
#  plt.show()ww
# NO outliers present in data set

# Checking skewness of the data
# for i in copied_data_set:
#  sns.histplot(copied_data_set[i])
#  plt.show()

# No skewness in data

# Checking Variation 
print(copied_data_set.var())

# High Variation so we will apply min max scalling
x_scaler=MinMaxScaler()
y_scaler=MinMaxScaler()

X = copied_data_set[['Hours']]  # 2D array
y = copied_data_set['Scores']  

x_scaled=x_scaler.fit_transform(X)
y_scaled=y_scaler.fit_transform(y.values.reshape(-1,1))
print(x_scaled)

# Trained and test split
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Intializing Instance of linear model
model=LinearRegression()
model.fit(X_train,y_train)

# Training the model 

# Now Tesiting the model
predicted_y=model.predict(X_test)
print("The Value of the predicted y is ",predicted_y)
# Checking the original value of the x_test
withoutscaled_x_test_value=x_scaler.inverse_transform(X_test)

print("The Value of the x_test without scaled is",withoutscaled_x_test_value)
without_scaled_predicted_y=y_scaler.inverse_transform(predicted_y)
print("The predicted Value of the y withoud scaled",without_scaled_predicted_y)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, predicted_y)
mse = mean_squared_error(y_test, predicted_y)

print(f"R2 Score: {r2}")
print(f"Mean Squared Error: {mse}")
