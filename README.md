# Wine-quality-classification
<br>
import pandas as pd
<br>
import numpy as np
<br>
import seaborn as sns
<br>
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
<br>
from sklearn.preprocessing import StandardScaler, LabelEncoder
<br>
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score)
<br>
from sklearn.linear_model import LogisticRegression
<br>
from xgboost import XGBClassifier
<br>
import warnings
****
warnings.filterwarnings('ignore')
def load_data_from_csv(file_path, target_column=None):
    try: 
        df = pd.read_csv(file_path)  
        print(df.columns)
        
        if target_column is None:
            common_targets = ['quality', 'target', 'class', 'label']  
            for col in common_targets:
                if col in df.columns:
                    target_column = col
                    break
            if target_column is None:
                target_column = df.columns[-1]  
                print(f'Target column : {target_column}')  # Fixed f-string
        
        # Separate features and targets
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target to numerical if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_names = le.classes_
            print(dict(enumerate(target_names)))
        else:
            target_names = [f"class_{i}" for i in np.unique(y)]
            print(f"Unique target values: {np.unique(y)}")
        
        return X, y, target_names, df, target_column
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None


# Exploratory data analysis

def perform_eda(df, target_column , target_names):
    
    missing_value = df.isnull().sum()
    print("Missing values is:{missing_value}") 

    if missing_value.sum()>0 :
   # Fill the numerical column with mean
      numerical_cols=df.select_dtype(include=[np.number]).columns
      df[numerical_cols]=df[numerical_cols].fillna(df[numerical_cols].mean())

   #Fill cathegorical column with mode
    categorical_cols=df.select_dtype(include=['object']).columns
    for col in categorical_cols:
     if col!=  target_column:
      df[col]=df[col].fillna(df[col].mode()[0])

      print(f"Missing data handled")
      

# Data preprocessing
def data_preprocessing(x,y):
   
   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2 , random_state=42)

  # Scale the feature

   scaler=StandardScaler()
   x_train_scaled = scaler.fit_transform(x_train)
   x_test_scaled = scaler.transform(x_test)

   return x_train_scaled,x_test_scaled,y_train,y_test,scaler

  #Model training evaluation
file_path="C:/Users/user/Downloads/Wine quality classification.csv"
x, y, target_names, df, target_column = load_data_from_csv(file_path)
df = perform_eda(df, target_column, target_names)

# Remap class label
le=LabelEncoder()
y_original=y.copy()
y=le.fit_transform(y)

x_train_scaled, x_test_scaled, y_train, y_test, scaler = data_preprocessing(x,y)

model = XGBClassifier(random_state=42, use_label_encoder= False , eval_metric='mlogloss')
model.fit(x_train_scaled, y_train)


y_pred=model.predict(x_test_scaled)
y_pred_prob=model.predict_proba(x_test_scaled)  




#calculate metrices
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average='weighted')
recall=recall_score(y_test,y_pred,average='weighted')
f1=f1_score(y_test,y_pred,average='weighted')
print(accuracy,precision,recall,f1)
 
# Cross validation Score
cv_scores=cross_val_score(model,x_train_scaled,y_train,cv=5)
cv_mean=cv_scores.mean()
print(cv_scores,cv_mean)


#save the model
import joblib
model_filename='wine quality classification'
scaler_filename='standard_scaler'
joblib.dump(model,model_filename)
joblib.dump(scaler,scaler_filename)
print("Model and scaler saved")
print(model_filename,scaler_filename)
