import numpy as np
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
x=np.arange(10).reshape(-1,1)
y=np.array([0,1,0,0,1,1,1,1,1,1])
model=LogisticRegression(solver="liblinear",C=10.0,random_state=0)
model.fit(x,y)
predictionprobability=model.predict_proba(x)
yprediction=model.predict(x)
score1=model.score(x,y)
confusion_matrix1=confusion_matrix(y,yprediction)
report=classification_report(y,yprediction)
print(f"X: {x}")
print(f"Y: {y}")
print(f"confusion_matrix1 : {confusion_matrix1}")
print(f" Report : {report}")
print(f"Y Prediction : {yprediction}")
print(f"Score : {score1}")
print(f" Model : {model}")
