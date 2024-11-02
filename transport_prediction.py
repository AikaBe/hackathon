import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import datetime
import itertools
import lightgbm as lgb
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters  import SimpleExpSmoothing
from statsmodels.tsa.holtwinters  import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import kagglehub
# Download latest version
path = kagglehub.dataset_download("serdargundogdu/municipality-bus-utilization")
print("Path to dataset files:", path)
df = pd.read_csv("municipality_bus_utilization.csv", parse_dates=['timestamp'])
df.head(20)
df.shape
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
df["timestamp"].min(), df["timestamp"].max()
df.info()
check_df(df)
df['municipality_id'].value_counts()
df.describe().T
sns.set_style("whitegrid")
plt.figure(figsize = (8, 4))
sns.barplot(x = df["municipality_id"], y = df["total_capacity"])
plt.xlabel('Municipality')
plt.ylabel('Total capacity')
plt.title('Total capacity per municipalities')
plt.show()
print("-"* 50)
capacities = df[["municipality_id", "total_capacity"]].drop_duplicates().sort_values("municipality_id")
for i in capacities.iterrows():
    print("Total capacity of the municipality {} = {} ~ {}%".format(
        i[1]["municipality_id"], i[1]["total_capacity"], round((i[1]["total_capacity"]*100)/sum(capacities["total_capacity"]), 2)))
   # print(i, type(i))
print("-"*50)
print("total capacity:", sum(capacities["total_capacity"]))
sns.set(style = 'whitegrid')
sns.FacetGrid(df, hue = 'municipality_id', height=6).map(sns.distplot, 'usage').add_legend()
plt.title('Distribution of Usages')
plt.show()
df6 = df[df.municipality_id == 6]
df6.head()
sns.set(style = 'whitegrid')
sns.FacetGrid(df6, hue = 'municipality_id', height=6).map(sns.distplot, 'usage').add_legend()
plt.title('Distribution of Usages')
plt.show()
plt.figure(figsize = (16, 4))
for i in range(10):
    plt.plot(df[df['municipality_id'] == i][['usage']].reset_index(drop=True), label=i)
plt.legend(loc='lower right',bbox_to_anchor=(1, 0.25))
plt.title('Usages in Time-series Format')
plt.show()
plt.figure(figsize = (16, 4))
plt.plot(df6[["timestamp"]], df6[['usage']].reset_index(drop=True), label=6)
plt.legend(loc='lower right',bbox_to_anchor=(1, 0.25))
plt.title('Usages in Time-series Format')
plt.show()
def create_date_features(df):
    df['hour'] = df.timestamp.dt.hour
    df['month'] = df.timestamp.dt.month
    df['day_of_month'] = df.timestamp.dt.day
    df['day_of_year'] = df.timestamp.dt.dayofyear
    df['week_of_year'] = df.timestamp.dt.weekofyear
    df['day_of_week'] = df.timestamp.dt.dayofweek
    df['year'] = df.timestamp.dt.year
    df["is_wknd"] = df.timestamp.dt.weekday // 4
    df['is_month_start'] = df.timestamp.dt.is_month_start.astype(int)
    df['is_month_end'] = df.timestamp.dt.is_month_end.astype(int)
    return df
df = create_date_features(df)
df.isnull().sum()
df.groupby(["municipality_id","hour"]).agg({"usage": ["count", "max"]})
df.groupby(["week_of_year","municipality_id",]).agg({"usage": ["count", "max"]})
df.head()
df_resampled = pd.DataFrame()
df["timestamp"] = df["timestamp"].astype(str).apply(lambda x: x[:-6]).astype("datetime64")
df_resampled = df.groupby(["timestamp","municipality_id"]).agg({"usage": "max"}).reset_index()
df_resampled.drop_duplicates(["timestamp","municipality_id"],inplace=True)
df_resampled.head()
dfs={}
for i in range(10):
    dfs[i]= pd.DataFrame(data=df_resampled[df_resampled.municipality_id==i], columns=["timestamp","usage"]).set_index("timestamp")
dfs[6].shape
type(dfs[6])
trains={}
tests={}
for i in range(10):
    trains[i] = dfs[i][:"2017-08-04 16:00:00"]
    tests[i] = dfs[i]["2017-08-05 07:00:00":]
    print (f"train {i} size:  ", len(trains[i]))
    print (f"test {i} size: ", len(tests[i]))
    trains[6].head()
tests[6].head()
def ses_optimizer(train,test, alphas, step=142):
    best_alpha, best_mae = None, float("inf")
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae
alphas = np.arange(0.01, 1, 0.10)
best_alpha, best_mae = ses_optimizer(trains[6],tests[6], alphas, step=142)
ses_model = SimpleExpSmoothing(trains[6]).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(142)
tests[6].head()
y_pred.reset_index(drop=True,inplace=True)
y_pred=pd.DataFrame(y_pred, columns=["usage"])
y_pred = y_pred.merge(tests[6].reset_index()["timestamp"], left_index=True, right_index=True).set_index("timestamp")
y_pred["usage"].head()
y_pred.index
trains[6]["usage"].head()
trains[6]["usage"].index
tests[6]["usage"].head()
tests[6]["usage"].index
def plot_prediction(i,y_pred, label):
    plt.figure(figsize=(16, 4))
    trains[i]["usage"].plot(legend=True, label=f"TRAIN {i}")
    tests[i]["usage"].plot(legend=True, label=f"TEST {i}")
    y_pred["usage"].plot(legend=True, label=f"PREDICTION {i}")
    plt.xlim([datetime.date(2017,6,4), datetime.date(2017,8,20)])
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()
plot_prediction(6, y_pred, "Single Exponential Smoothing")
def des_optimizer(train,test, alphas, betas, step=142):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae
alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)
best_alpha, best_beta, best_mae = des_optimizer(trains[6],tests[6], alphas, betas, step=142)
alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)
best_alpha, best_beta, best_mae = des_optimizer(trains[6],tests[6], alphas, betas, step=142)
des_model = ExponentialSmoothing(trains[6], trend="add").fit(smoothing_level=best_alpha,
smoothing_slope=best_beta)
y_pred = des_model.forecast(142)
y_pred.reset_index(drop=True,inplace=True)
y_pred=pd.DataFrame(y_pred, columns=["usage"])
y_pred = y_pred.merge(tests[6].reset_index()["timestamp"], left_index=True, right_index=True).set_index("timestamp")
y_pred["usage"].head()
plot_prediction(6, y_pred, "Double Exponential Smoothing")
def tes_optimizer(train,test, abg, step=142):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=10).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_gamma, best_mae
alphas = betas = gammas = np.arange(0.01, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))
best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(trains[6], tests[6], abg, step=142)
y_pred.head()
y_pred.reset_index(drop=True,inplace=True)
y_pred=pd.DataFrame(y_pred, columns=["usage"])
y_pred = y_pred.merge(tests[6].reset_index()["timestamp"], left_index=True, right_index=True).set_index("timestamp")
y_pred["usage"].head()
plot_prediction(6, y_pred, "Triple Exponential Smoothing ADD")
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]
def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}4 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}4 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order
best_order, best_seasonal_order = sarima_optimizer_aic(trains[6], pdq, seasonal_pdq)
#Random NOISE 
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))
df.head()
df.sort_values(by=["municipality_id", "total_capacity","timestamp"], axis=0, inplace=True)
df.head()
pd.DataFrame({"usage": df["usage"].values[0:10],
              "lag1": df["usage"].shift(1).values[0:10],
              "lag2": df["usage"].shift(2).values[0:10],
              "lag3": df["usage"].shift(3).values[0:10],
              "lag4": df["usage"].shift(4).values[0:10]})
df.groupby(["municipality_id","total_capacity"])["usage"].head()
df.groupby(["municipality_id","total_capacity"])["usage"].transform(lambda x: x.shift(1))
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['usage_lag_' + str(lag)] = dataframe.groupby(["municipality_id", "total_capacity"])['usage'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
check_df(df)
pd.DataFrame({"usage": df["usage"].values[0:10],
              "roll2": df["usage"].rolling(window=2).mean().values[0:10],
              "roll3": df["usage"].rolling(window=3).mean().values[0:10],
              "roll5": df["usage"].rolling(window=5).mean().values[0:10]})
pd.DataFrame({"usage": df["usage"].values[0:10],
              "roll2": df["usage"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["usage"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["usage"].shift(1).rolling(window=5).mean().values[0:10]})
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['usage_roll_mean_' + str(window)] = dataframe.groupby(["municipality_id", "total_capacity"])['usage']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe
df = roll_mean_features(df, [365, 546])
df.head()
pd.DataFrame({"usage": df["usage"].values[0:10],
              "roll2": df["usage"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["usage"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["usage"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["usage"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["usage"].shift(1).ewm(alpha=0.1).mean().values[0:10]})
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['usage_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["municipality_id", "total_capacity"])['usage'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe
alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
df = ewm_features(df, alphas, lags)
check_df(df)
df = pd.get_dummies(df, columns=['municipality_id', 'total_capacity', 'day_of_week', 'hour'])
check_df(df)
train = df.loc[(df["timestamp"] <"2017-08-05"),:]
val = df.loc[(df["timestamp"] >="2017-08-05"),:]
cols = [col for col in train.columns if col not in ["timestamp", "usage","year"]]
Y_train = train["usage"]
X_train = train[cols]
Y_val = val['usage']
X_val = val[cols]
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape
lgb_params = {
    'num_leaves': 10,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'max_depth': 5,
    'verbose': 0,
    'num_boost_round': 1000,
    'early_stopping_rounds': 200,
    'nthread': -1
}
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  verbose_eval=100)
print(lgb.__version__)
# Завершение функции для SARIMAX
def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('Best SARIMA{}x{} - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

# Оптимизация SARIMA
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 10) for x in list(itertools.product(p, d, q))]  # Измените 4 на 10, если у вас 10 периодов
best_order, best_seasonal_order = sarima_optimizer_aic(trains[6], pdq, seasonal_pdq)

# Прогнозирование с использованием найденных параметров
sarima_model = SARIMAX(trains[6], order=best_order, seasonal_order=best_seasonal_order).fit(disp=0)
y_pred = sarima_model.forecast(steps=142)  # 142 - количество шагов в будущее

# Подготовка y_pred для визуализации
y_pred = pd.DataFrame(y_pred, columns=["usage"])
y_pred = y_pred.merge(tests[6].reset_index()["timestamp"], left_index=True, right_index=True).set_index("timestamp")

# Визуализация прогноза
plot_prediction(6, y_pred, "SARIMA Forecast")
