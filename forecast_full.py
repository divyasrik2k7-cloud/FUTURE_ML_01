# forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# -----------------------------
# Step 1: Load your dataset
# -----------------------------
# Ensure monthly_sales.csv has two columns: ds (date), y (sales)
df = pd.read_csv("monthly_sales.csv")
df['ds'] = pd.to_datetime(df['ds'])   # convert to datetime

print("Data loaded ✅")
print(df.head())

# -----------------------------
# Step 2: Visualize historical sales
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(df['ds'], df['y'], marker='o')
plt.title("Historical Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# -----------------------------
# Step 3: Train/Test split
# -----------------------------
# Use last 12 months as test set
train = df.iloc[:-12]
test = df.iloc[-12:]

# -----------------------------
# Step 4: Build and fit Prophet model
# -----------------------------
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(train)

# -----------------------------
# Step 5: Forecast future
# -----------------------------
future = model.make_future_dataframe(periods=12, freq="MS")  # next 12 months
forecast = model.predict(future)

# -----------------------------
# Step 6: Evaluate accuracy
# -----------------------------
pred = forecast.set_index("ds").loc[test["ds"], "yhat"]

mape = np.mean(np.abs((test["y"].values - pred.values) / test["y"].values)) * 100
print(f"MAPE on test set: {mape:.2f}%")

# -----------------------------
# Step 7: Plot forecast
# -----------------------------
model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

model.plot_components(forecast)
plt.show()

# -----------------------------
# Step 8: Export forecast for Power BI
# -----------------------------
forecast_out = forecast[['ds','yhat','yhat_lower','yhat_upper']]
forecast_out.to_csv("sales_forecast.csv", index=False)
print("Forecast exported to sales_forecast.csv ✅")