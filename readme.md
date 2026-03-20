PROJECT EXPLANATION

OVERVIEW:
This project builds a machine learning system that predicts what the maximum temeprature will be tomorrow by reading few days of weather data of Delhi.
This is called regression problem : the output is a continous number.
The system processes 4 yrs of daily weather data observations, From each day's data it creates a set of input features, then train three differernt models to learn the pattern between those features and the next day's temperature.

===================================================

Data is taken from open-meteo.com
Three models used are : (1) Linear Regression, (2) Ridge Regression, (3) Random Forest

===================================================

PROJECT FILE STRUCTURE
dataset.py: This file produce a clean, feature-rich dataset and save it as delhi_weather.csv. It tries to download real data from the Open-meteo API first. If there is no internet connection, it generates realistic synthetic data based on Delhi's known soasonal statistic. Every way, the output format is identical.

main.py: This is the main pipeline file. It loads the dataset, splits it into training and test sets, scales the features, train three modles, evaluates them on the test set, generates six plots, and prints a formatted summary table.

===================================================

LIBRARIES USED

(3.1) numpy: NumPy(Numerical Python) is python's core library for numerical computation. It provides the n dimensional array that is much faster than a regular lists for mathematical operations.
Where it is used in this project?
(a) numpy.random == generates randon numbers for the synthetic data generator like random temp, rainfall amounts, wind speed.
(b) numpy.random.seed == sets a fixed random seed so the generated data is identical every time we run the program.
(c) numpy.argsort == sort feature importances from highest to lowest in the importance analysis feature.
(d) numpy.sqrt == computes the square root, used for calculating root mean squared error from mean squared error.
(e) numpy.cos , numpy.sin, numpy.pi == used for temperature formula inside generate_delhi_data() to create the sinusoidal temperature cycle.

(3.2) pandas : pandas is python's library for working with tabular data, like an excel sheet inside python. 
Where it is used in this project?
(a) pd.DataFrame == creates the main dataset table from lists of weather values.
(b) pd.data_range == generates a sequence of daily dates.
(c) pd.to_datetime == converts date strings which we got from the api, into pandas timestamp objects that can be sortes, compared, and queried.
(d) pd.read_csv / df,to_csv == reads and write the dataset to/from a csv file so we don't regenerate it every run.
(e) df['column'].shift(n) == shifts a column down by n rows, creating lag features (yesterday's value
becomes a new column for today's row).
(f) df['column'].rolling(7).mean() == computes a 7-day rolling average , the mean of the current and
previous 6 values.
(g) df.dropna() == removes rows that have missing values (the first few rows have no lag data available,
so we drop them).
(h) df.groupby('month')['temp_max'].mean() — groups data by month and computes the average
temperature for each month (used in Plot 1).
(i) df.describe() == computes summary statistics, it count, mean, std, min, max, percentiles for all numeric values.

(3.3) matplotlib == It's a python library which is used to create virtually any type of chart(line plots, bar charts, scatter plots, histograms, etc.)
Where it is used in this project? 
(a) plt.subplots(rows, cols, figsize) == create a figure with one or more subplot panels, ex - plt.subplots(3,1) creates 3 stacked panels used in plot 3.
(b) ax.plot(x,y) == draws a line connecting data points, used for temperature time series and actual vs predicted lines.
(c) ax.fill_between(x, y1, y2) == fills the ares between two lines with a transparent color, used to highlight prediction errors.
(d) ax.bar / ax.brah = vertical and horizontal bar chats, used in the model comparison plot and feature importance plot.
(e) ax.scatter(x,y) == scatter plot of predicted vs actual temperatures.
(f) plt.rcParams.update == sets global settings (font, DPI, border removal) so all plots have a consistent, clean appearence.
(g) plt.savefig(path) == saves each finished plot as a PNG file to the results/folder.
(h) plt.close(fig) == close the figure after saving to free memory.

(3.4) seaborn : It is built on top of matplotlib and provides higher level statistical visualisation functions with better default aesthetics.

(3.5) scikit-learn : it is pythons machine learning library. It provides consistent, well-tested implementations of ml algos, every model has .fit(), .predict(), .score() methods.
Where it is used in project?
(a) Linear Regression(from sklearn.linear_model) == implements ordinary leaast squares linear regression.
(b) Ridge == implements L2- regularised linear regression, same interface as linearregression but adda a penalty term to prevent overfitting.
(c) Random Forest == builds 200 decisions trees in llal, each on a random subset of data and averages their predictions.
(d) standardscaler == Transform features to zero mean and unit variance. .fit_transform(X_train) learns statistics from traing data and applies tehm. .transform(X_test) applies the same learned statistics to test data.
(e) mean_absolute_error == Computes the average of(|actual - predicted|) across all test samples.
(f) mean_squared_error == computes the average of (|actual - predicted|)^2, we then take sqrt() to get RMSE.
(g) r2_score = computes R-squared , the fraction of variance in the target explained by the model.

(3.6) os : Python's library to interact with the operating system, file paths, directory creation, etc.

(3.7) requests : pyhton's library for making http web requests, fetching data from urls, calling APIs.
Where it is used in this project?
In dataset.py, request.get(url, params) calls open-meteo api to download real delhi weather data. The params dictionary is automatically converted to url query parameters. The response arrives as JSON, which we convert to a python dictonary with response.json

(3.8) warnings : pythons library for controlling warning messages
Where it is used in this project? 
warnings.filterwarnings('ignore') supresses minor sklearn warnings such as aonvergaence warnings on small datasets, that would otherwise clutter the terminal output without affecting resuls.

===================================================

dataset.py-- explanation

It contains three function to produce the final dataset.csv file

(4.1) download_real_data() == This function calls the open-meteo historical weather api for delhi, Open-meteo is a free, open-source weather api the provides ERA5 reanalysis data, a globally gridded reconstruction of historical weather based on assimilating billions of oberservations into numerical weather model.
It has 4 daily variables:
(i) temperature_2m_max == highest temp recorded at 2 meter above the ground in 24 hours period.
(ii) temperature_2m_min == the lowest temp recorded at 2 meter above the ground in 24 hours period.
(iii) precipitation_sum == total rainfall accumulated in that 24-hour period(mm).
(iv) windspeed_10m_max == the highest 10-min average wind speed at 10m height recorded that day(km/h).
The API return back as json, we extract the 'daily' section and build a pandas dataframe from it. If anything falis(no internet, apid down, network timeout), the function returns none and programs falls back to generated data.

(4.2) generate_delhi_data() == When real data is unavailable, this function generates days of synthetic weather that statistically matches Delhi's observed climate.
Genertation uses:
(i) Sinusoidal seasonal cycle : delhi's average temperature follows a roughly cosine-shaped curve through the year. The formula used is: seasonal_base = 29.5 - 12.5 * cos(2*pi*(day_of_year - 15)/365)
(ii) Random noise : Real weather is never perfectly smooth. We add np.random.normal(0, 2.5) to each day, a random perturbation with mean 0 and std 2.5'C. This creates realistic day-to-day variation around the seasonal trend.
(iii) Monsoon rainfall model : Delhi receives 85% of its annual rainfall during June-September. During these months, each day has 60% probability of rain, with amounts drawn from an exponential distribution (np.random.exponential(12)) whixh naturally produces mostly light rain events with occasional heavy events, matching observed rainfall statistics.
Monsoon cooling : Heavy rainfall days are 1-4'C cooler than non-rainy days due to cloud cover and evaporative cooling. This is modelled by subtracting a random value from tmax when rainfall exceeds 5mm.
Loo winds : April-June in delhi is notorious for the 'loo'- hot, dry wind. The wind speed model assign 15-35 ikm/h during these months versus 5-20 km/hr for other months.

(4.3) engineer_features() == Raw weather data(date, temp_max, temp_min, rainfal, windspeed) is not directly useful for prediction. We need to transform it into features that captures the patterns a model can learn from.

===================================================

(5) Feature engineering : It captures new input columns from raw data that make it easier for a model to find patters. It is often the most impactful step in an ML project, better features beat better algoritms.
(i) Lag feature == It is the value  of a variable from a previous time step, added as a new column for the current row.
df['temp_max_lag1'] = df['temp_max'].shift(1) # yesterday's max temp
df['temp_max_lag2'] = df['temp_max'].shift(2) # two days ago
df['temp_max_lag3'] = df['temp_max'].shift(3) # three days ago
df['temp_min_lag1'] = df['temp_min'].shift(1) # yesterday's min temp
This .shift(n) method in pandas moves every values in a column dawn by n rows, so that row i(today) now contains the value that was originally in row i-n(n days).
Physical justification: Weather has inertia. If today is 40°C, tomorrow is very unlikely to be 20°C. The temperature from the past 2–3 days is the single strongest predictor of tomorrow's temperature. This is confirmed by the feature importance analysis, which shows lag features ranking highly.

(5.2) Rolling Mean(7-day average)
df['temp_rolling_7d'] = df['temp_max'].shift(1).rolling(window=7,
min_periods=7).mean()
The 7-day rolling mean is the average of the past 7 days maximum temperatures. We shift by 1 first to ensure we only use past information (not today's valuse), then compute the rolling average over the preceding 7 days.

(5.3) Temperature Range == df['temp_range_lag1'] = df['temp_max_lag1'] - df['temp_min_lag1']
The diurnal (daily) temperature range is the difference between the day's maximum and minimum
temperature. A large range (e.g. 15°C) indicates a clear, sunny day with dry air. A small range (e.g. 5°C) indicates cloudy or rainy conditions that moderate both the high and the low.

(5.4) Rainfall Lag == df['rainfall_lag1'] = df['rainfall'].shift(1)
Yesterday's total rainfall. Heavy rain days are typically followed by cooler temperatures due to soil moisture and residual cloud cover. This feature captures that post-rain cooling effect.

(5.5) Mounth and Day of year == 
df['month'] = df['date'].dt.month # integer 1-12
df['day_of_year'] = df['date'].dt.day_of_year # integer 1-365
Month captures coarse seasonality: January is always cold, May is always hot. Day of year is more precise it allows the model to distinguish early May (temperature still rising) from late May (approaching peak). The .dt accessor in pandas extracts date components from a datetime column.

===================================================

main.py : It runs the full ML experiment from start to finish. Here is what every major step does and why it is done in that order.

(i) Load Data == If delhi_weather.csv exists from a previous run, load it directly. Otherwise call get_dataset() from dataset.py which generates the csv. This avoids re-running the API call or synthetic genertation every time.
(ii) Explore the data == Before any modelling, print basic statistics about the dataseet: total rows, date range, min/max temperatures, number of rainy days. This is called Exploratory Data Analysis(EDA). It confirms the data looks sensible, delhi's hottest day is 47.8'C(plausible) and coldest is 10.7'C.