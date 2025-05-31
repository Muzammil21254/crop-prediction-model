import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
data=pd.read_csv("Crop_recommendation.csv")
data.head()
data.shape
data.isnull().sum()
x=data.iloc[:,:-1] # features
y=data.iloc[:,-1] # labels
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
model=RandomForestClassifier()
print(model)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
accuracy=model.score(x_test,y_test)
print("accuracy:",accuracy)
new_features=[[36,58,25,28.66024,59.31891,8.399136,36.9263]]
predicted_crop=model.predict(new_features)
print("predicted crop:",predicted_crop)
pickle.dump(model,open("model.pkl","wb"))