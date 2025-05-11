import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
lis_of_custom_null=['null', 'na', 'n/a', 'N/A', 'missing', '_', '', '-','none', 'nan',
    'Not Applicable', 'Unknown', 'Unspecified', '?', '--', 'null value', 'no data'
]
original_data=pd.read_csv("C:/Users/PMLS/Downloads/diabetes.csv",na_values=lis_of_custom_null)

copied_data=original_data.copy()
print(copied_data.isnull().sum())
# No null values
# checing negative values
# Describe
print(copied_data.describe())
# detecting outliers
# for i in copied_data:
#    sns.boxplot(copied_data[i])
#    plt.show()

# Description
# pregnancies has 3 outliers but outliers are realistics so we cant remove it


col_has_zer_un_realistic=copied_data[['Glucose', 'BloodPressure', 'SkinThickness','BMI','DiabetesPedigreeFunction']]
col_has_nonrealistic=col_has_zer_un_realistic.replace(0,np.nan)

print(f" After it replaciing with nan the null  values",col_has_nonrealistic.isnull().sum())

# Again checking the outliers of these columns

# Refelecting the changes to the original data set
copied_data[['Glucose', 'BloodPressure', 'SkinThickness','BMI','DiabetesPedigreeFunction']]=col_has_nonrealistic
# Glucose column has no outliers but the blood pressure , skin thickness , insulin too much ,BMI,diabeties predegree function 


print("The Value of the SkinThickness ",copied_data['SkinThickness'].isnull().sum())
# checking skewness of the data 
print("checkingg the describe of  copied_data the column after reflecting changes into it ",copied_data.describe())
for i in copied_data:
    print(f"The Skewness {i} of the data before logged transformation",copied_data[i].skew())

# applying log transformation to the columns
# for_log=copied_data.drop(columns='Outcome')
# logged_trans_col=np.log1p(for_log)



# for i in logged_trans_col:
#    print(f"Again checking the skewness of the data after logged transformation",logged_trans_col[i].skew())

# when we applying log transformation data has been left skewes to remove outliers we will used IQR method

features=copied_data.drop(columns='Outcome')
target=copied_data['Outcome']
print("checking the null values of it before detecting the outliers ",copied_data.isnull().sum())
for i in features:
    Q1=features[i].quantile(0.25)
    Q3=features[i].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upperbound=Q3+1.5*IQR
    if(i =='Pregnancies'):
        lower_bound=max(0,lower_bound)
        upperbound=max(17,upperbound)
    elif (i=='Insulin'):
        lower_bound=max(0,lower_bound)
    elif(i=='Age'):
        lower_bound=max(5,lower_bound)
    elif(i=='DiabetesPedigreeFunction'):
        lower_bound=max(0,lower_bound)
    
    
    print(f" lower bound of that {i}",lower_bound)
    print(f"The upper bound of that {i}",upperbound)
# Managed bounds of all the column now handling outliers
    
    outliers=(features[i] < lower_bound) | (features[i]>upperbound)
    print(f"The Outliers {i} :",features[i][outliers])
    copied_data.loc[outliers, i] = np.nan


# after replacing it with outliers checking all the columns that has null values in it
print("NUll values present in all columns are",copied_data.isnull().sum())

# Data are right skewes applying log transformation on it

# checking the skewness again after handled the outliers and replace it with
for i in copied_data:
   print(f"The Skewness of the {i}", skew(copied_data[i],nan_policy='omit'))


# for half of the column we will fill values with mean
# and the rest half will filled with median


print(copied_data.head(5)
      )
print(original_data.head(5))
copied_data['Pregnancies']=copied_data['Pregnancies'].fillna(copied_data['Pregnancies'].median())
copied_data['SkinThickness']=copied_data['SkinThickness'].fillna(copied_data['SkinThickness'].mean())
copied_data['Glucose']=copied_data['Glucose'].fillna(copied_data['Glucose'].median())
copied_data['BloodPressure']=copied_data['BloodPressure'].fillna(copied_data['BloodPressure'].mean())

print("The Lower Bound of the insulin is ",copied_data['Insulin'].min())
print("The Null values in the coopied data set are",copied_data['Insulin'].isnull().sum())
print("The Null values of the insulin column in the original data set are ",original_data['Insulin'].isnull().sum())


copied_data['Insulin']=copied_data['Insulin'].fillna(copied_data['Insulin'].mean())
copied_data['BMI']=copied_data['BMI'].fillna(copied_data['BMI'].mean())
copied_data['DiabetesPedigreeFunction']=copied_data['DiabetesPedigreeFunction'].fillna(copied_data['DiabetesPedigreeFunction'].median())
copied_data['Age']=copied_data['Age'].fillna(copied_data['Age'].median())

# Again CHecking the null values are in the copied data set
print("Null values in the copied data sets are",copied_data.isnull().sum())

print("The min max after remove outliers",copied_data['Age'].min())

# Again Checking the skewness of all data 
for i in copied_data:
    print(f"The Skewness of the {i} is ",copied_data[i].skew())

# Half of the columns are tightly right skewed
# Applying log transformation on it 

print("The column of the copied data sets are",copied_data.columns)
features=copied_data.drop('Outcome',axis=1)
target=copied_data['Outcome']




# scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# # Check data types
# for col in scaled_df.columns:
#     print(f"The Data type of the {col} is", scaled_df[col].dtype)

# # Check skewness
# for col in scaled_df.columns:
#     print(f"The Skewness of the scaled {col} is", skew(scaled_df[col]))

# # print(scaled_features)


# Before training the model checking the correlation of features with target
print(copied_data.corr())



for i in copied_data:
    print(f"The Min , Max of {i} are",copied_data[i].describe())





# will apply log transformation on some columns
copied_data['Pregnancies']=np.log1p(copied_data['Pregnancies']
                                    )
copied_data['Glucose']=np.log1p(copied_data['Glucose'])
copied_data['DiabetesPedigreeFunction']=np.log1p(copied_data['DiabetesPedigreeFunction'])
copied_data['Insulin']=np.log1p(copied_data['Insulin']
                                )

copied_data['Age']=np.log1p(copied_data['Age']
                            )


# Again checking the skewness of the data 



from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

model_x_features=copied_data.drop(columns='Outcome')
y_target=copied_data['Outcome']

x_scaler=MinMaxScaler()
y_scaler=MinMaxScaler()
scaled_features=x_scaler.fit_transform(model_x_features)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, y_target, test_size=0.2, random_state=42)


model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
predicted_y=model.predict(X_test)
print(predicted_y)




# Saving MOdel Before Saving it checking min max of all the columns
import joblib

joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(x_scaler,"features scaler.pkl")
# WIll apply transformation on Age Insulin Glucose Pregnancies abetesPedigreeFunction
tranformation_col = ['Pregnancies', 'Glucose', 'Age', 'Insulin', 'DiabetesPedigreeFunction']


# Loading the model 

loaded_model = joblib.load("logistic_regression_model.pkl")
load_x_scaler = joblib.load("features scaler.pkl")

# Taking Input from the user
print("The Columns",copied_data.columns)

# Getting input from the user
# Sample bounds as per your message (replace with actual IQR-calculated values if needed)
bounds = {
    'Pregnancies': (0, 17),
    'Glucose': (36, 204),
    'BloodPressure': (40, 104),
    'SkinThickness': (1.0, 57),
    'Insulin': (0, 318),
    'BMI': (13, 50),
    'DiabetesPedigreeFunction': (0, 1.2),
    'Age': (5, 66.5)
}

# Taking input and validating
pregnancies = int(input(f"Enter the value for Pregnancies {bounds['Pregnancies']}: "))
if pregnancies < bounds['Pregnancies'][0] or pregnancies >bounds['Pregnancies'][1]:
    print("Pregnancies value is out of bound.")
    exit()

Glucose = float(input(f"Enter the value for Glucose {bounds['Glucose']}: "))
if Glucose < bounds['Glucose'][0] or Glucose > bounds['Glucose'][1]:
    print("Glucose value is out of bound.")
    exit()

BloodPressure = float(input(f"Enter the value for BloodPressure {bounds['BloodPressure']}: "))
if BloodPressure < bounds['BloodPressure'][0] or BloodPressure > bounds['BloodPressure'][1]:
    print("BloodPressure value is out of bound.")
    exit()

SkinThickness = float(input(f"Enter the value for SkinThickness {bounds['SkinThickness']}: "))
if SkinThickness < bounds['SkinThickness'][0] or SkinThickness > bounds['SkinThickness'][1]:
    print("SkinThickness value is out of bound.")
    exit()

Insulin = float(input(f"Enter the value for Insulin {bounds['Insulin']}: "))
if Insulin < bounds['Insulin'][0] or Insulin > bounds['Insulin'][1]:
    print("Insulin value is out of bound.")
    exit()

BMI = float(input(f"Enter the value for BMI {bounds['BMI']}: "))
if BMI < bounds['BMI'][0] or BMI > bounds['BMI'][1]:
    print("BMI value is out of bound.")
    exit()

DiabetesPedigreeFunction = float(input(f"Enter the value for DiabetesPedigreeFunction {bounds['DiabetesPedigreeFunction']}: "))
if DiabetesPedigreeFunction < bounds['DiabetesPedigreeFunction'][0] or DiabetesPedigreeFunction > bounds['DiabetesPedigreeFunction'][1]:
    print("DiabetesPedigreeFunction value is out of bound.")
    exit()

Age = int(input(f"Enter the value for Age {bounds['Age']}: "))
if Age < bounds['Age'][0] or Age > bounds['Age'][1]:
    print("Age value is out of bound.")
    exit()

# Creating the DataFrame
import pandas as pd
data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age]
})

print("\nUser input is valid. Here's the DataFrame:")
print(data)




# Apply log transformation to the selected columns
log_transformed_columns = ['Pregnancies', 'Glucose', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
for col in log_transformed_columns:
    data[col] = np.log1p(data[col])

# Scale the features using the previously fitted MinMaxScaler
scaled_features = load_x_scaler.transform(data)

# Make predictions
prediction = loaded_model.predict(scaled_features)

# Output the prediction
if prediction[0] == 0:
    print("The prediction is: No Diabetes")
else:
    print("The prediction is: Diabetes")