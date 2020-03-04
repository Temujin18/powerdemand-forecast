import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('minda_power.csv')
# print(df)

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=120, freq='M')
# print(future.tail())

forecast = m.predict(future)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

m.plot(forecast)
# m.plot_components(forecast)
plt.show()