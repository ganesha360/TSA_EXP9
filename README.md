# EX-NO: 09 - A project on Time series analysis on Yahoo stock prediction using the ARIMA model 

### Name: GANESH R
### Register No: 212222240029
### Date: 

## AIM:
To create a project on time series analysis for **Yahoo stock prediction** using the ARIMA model in Python and evaluate its performance.

## ALGORITHM:
1. Load and explore the stock dataset.
2. Check for stationarity of the time series using:
   - Time series plot
   - ACF plot
   - PACF plot
   - ADF test
3. Transform to stationary:
4. Differencing
5. Determine ARIMA model parameters p, d, q.
6. Fit the ARIMA model to the stock data.
7. Make predictions for future stock prices.
8. Evaluate model predictions against actual values using metrics like Mean Squared Error.
9. Visualize the predicted and actual stock prices.
    
## PROGRAM:

**1. Import Necessary Libraries:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
```

**2. Load the Dataset:**
```python
df = pd.read_csv('/content/yahoo_stock.csv', parse_dates=['Date'], index_col='Date')
df.dropna(inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.head()
```

**3. Plot the Time Series Data:**
```python
train_size = int(len(df) * 0.8)
train_data, test_data = df['Log_Close_diff'][:train_size], df['Log_Close_diff'][train_size:]

plt.figure(figsize=(10, 6))
plt.plot(df['Close'])
plt.title('Yahoo Stock Price')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()
```

**4. Check for Stationarity with ADF Test:**
```python
def test_stationarity(timeseries):
    adf_result = adfuller(timeseries)
    print('ADF Statistic:', adf_result[0])
    print('p-value:', adf_result[1])
    
    if adf_result[1] <= 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is non-stationary.")

test_stationarity(df['Close'])
```

**5. Visualize ACF and PACF Plots:**
```python
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
plot_acf(df['Close'], lags=30, ax=ax[0])
plot_pacf(df['Close'], lags=30, ax=ax[1])
plt.show()
```

**6. Transformation:**
```python
df['Close_diff'] = df['Close'].diff()
df['Log_Close'] = np.log(df['Close'])
df['Log_Close_diff'] = df['Log_Close'].diff()

plt.figure(figsize=(12, 6))
plt.plot(df['Close_diff'], label='Differenced Close Price')
plt.title('Differenced Stock Price')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.legend()
plt.show()
```

**7. Model Implementation:**
```python
p = 1
d = 1
q = 1

model = ARIMA(df['Close'], order=(p, d, q))
fit_model = model.fit()
print(fit_model.summary())

forecast = fit_model.forecast(steps=30)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
forecast_series = pd.Series(forecast, index=forecast_index)

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Historical Close Price')
plt.plot(forecast_series, label='Forecasted Close Price', color='orange')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

predictions = fit_model.forecast(steps=len(test_data))

mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error:', mse)
```

## OUTPUT:

### Original Dataset and Graphical Representation:
| Image |
|-------|
| ![Original Dataset](https://github.com/user-attachments/assets/e58bd28f-be3c-438e-86c5-f68ea5b974fb) |
| ![Stock Price Plot](https://github.com/user-attachments/assets/8404d105-768c-4e11-92b5-73e43ac8967b) |

### Autocorrelation and Partial Autocorrelation Representation:
| ACF Plot and PACF Plot| 
|----------|
| ![ACF Plot](https://github.com/user-attachments/assets/eb586082-9971-4468-80ff-8b92b0b9c85e) |

### Differenced Representation:
| Image |
|-------|
| ![Differenced Stock Price](https://github.com/user-attachments/assets/3aff53c6-0c1e-4c30-aca4-a40bbf87734a) |

### Model Results:
| Image |
|-------|
| ![Model Summary](https://github.com/user-attachments/assets/dfcd93a1-eab9-4834-a6f9-a1e79b97a5e5) |

### Forecasted Representation:
| Image |
|-------|
| ![Forecasted Stock Price](https://github.com/user-attachments/assets/56a612ab-c71e-45d5-bf55-2bc4c4ec6b47) |

### Model Results Summary:
| Image |
|-------|
| ![Model Results](https://github.com/user-attachments/assets/b9752e64-d73a-48bb-8997-dd01c296d8e3) |

## RESULT:
Thus, the program ran successfully based on the ARIMA model using Python.
