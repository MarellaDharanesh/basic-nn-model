# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

In this model we will discuss with a neural network with 3 layers of neurons excluding input . First hidden layer with 3 neurons , Second hidden layer with 17 neurons and final Output layer with 1 neuron to predict the regression case scenario.

we use the relationship between input and output which is 
output = input * 17
and using epoch of about 1000 to train and test the model and finnaly predicting the  output for unseen test case.

## Neural Network Model
![262085896-0a19cc22-3c90-4158-9373-4f8fde080468](https://github.com/SASIRAJ27/basic-nn-model/assets/113497176/5dfaea7f-c76e-4f79-98e5-306ef49ee119)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Marella Dharanesh
### register number 212222240062
```python
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dlexp1').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
X = df[['input']].values
y = df[['output']].values

X


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai=Sequential([
    Dense(3,activation='relu',input_shape=[1]),
    Dense(17,activation='relu'),
    Dense(1)
])

ai.compile(optimizer='rmsprop',loss='mse')

ai.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)

z = [[16]]
z1=Scaler.transform(z)
ai.predict(z1)

```

## Dataset Information
![image](https://github.com/MarellaDharanesh/basic-nn-model/assets/118707669/9a691b15-0fed-4d00-9a73-0717a1fceb82)


## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://github.com/MarellaDharanesh/basic-nn-model/assets/118707669/9ccfda19-c08b-4112-b105-64da032051d3)



### Test Data Root Mean Squared Error

![image](https://github.com/MarellaDharanesh/basic-nn-model/assets/118707669/5c1dd915-d06e-41c5-a0b5-c8c75fc41805)
![image](https://github.com/MarellaDharanesh/basic-nn-model/assets/118707669/4cd56fc9-8da0-4649-b12c-99afe9ddd18a)



### New Sample Data Prediction
![image](https://github.com/MarellaDharanesh/basic-nn-model/assets/118707669/36eb0713-bdd2-45d6-a547-d7e7b5e57934)


## RESULT

Thus a neural network regression model for the given dataset is written and executed successfully.
