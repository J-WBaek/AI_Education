import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, RNN, LSTM, Dropout

# 범위를 0 ~ 1 로 normalized 
def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)


df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

dfx.describe()

# 두 데이터를 리스트 형태로 저장
x = dfx.values.tolist() # open, high, log, volume, 데이터
y = dfy.values.tolist() # close 데이터

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x,  label = {'Open','High','Low','Volume'})
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(y, label ='Close')
plt.legend()
plt.show()

# 6월 1일 ~ 6월 10일까지의 OHLV 데이터로 6월 11일 종가 예측
# 6월 2일 ~ 6월 11일까지의 OHLV 데이터로 6월 12일 종가 예측
window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)



train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])

print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', 
              loss='mean_squared_error')
history = model.fit(train_x, train_y, 
                    validation_data = (val_x, val_y), 
                    epochs=70, batch_size=30)


with open('history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
model.save('GoogleNet_mnist.h5')

pred_y = model.predict(test_x)

plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()