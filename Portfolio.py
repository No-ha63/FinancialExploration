import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import yfinance as yf
import math as m
from scipy.stats import norm
from scipy.optimize import minimize

total = 3 * 365


def get_data(tickers : list, days = total):
    '''Inputs: tickers -> list of stocks for portfolio ; days -> amount for stock's history.
    Returns a dictionary of the stocks with the necessary data for Monte Carlo Simulations'''
    end = dict()
    closes = dict()
    date_start = dt.date.today() - dt.timedelta(days = days)
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        data = stock.history(start = date_start)
        data = data.drop(columns=['Volume','Dividends','Stock Splits','Open','High','Low']) 
        data["Percent Change"] = data["Close"].pct_change() 
        data["Log Change"] = np.log(data['Percent Change'] + 1)
        end[ticker] = data
        closes[ticker] = data['Close'][-1]
    return end, closes

def GBM(data, num = 10000, days = 252, window = total):
    if window == total:
        window = 3 * 252
    log_change = data["Log Change"][- window : ]
    mu = np.mean(log_change)
    std = np.std(log_change)
    drift = mu - (1/2) * std**2

    # X = mu + sigma * Z where Z is N(0,1) gives N(mu,sigma^2)
    #delta_t = 1 as using daily increments and data for drift and volatility are daily closings
    z_updated = np.exp(np.random.normal(loc = drift, scale = std, size = (days,num))) 
    end = np.zeros(shape=(days+1,num))
    end[0] = data['Close'][-1]
    for i in range(1,days+1):
        end[i] = end[i-1] * z_updated[i-1] #Multiplying lognormal by lognormal produces lognormal
    return end[-1]

def Port_GBM(data: dict, weights: np.ndarray, num = 10000, days = 252, intiial_value = 1000):
    percent_changes_df = pd.DataFrame(index=data[list(data.keys())[0]].index)
    for ticker in data:
        percent_changes_df[ticker] = data[ticker]['Percent Change']
    
    means = percent_changes_df.mean()
    covMatrix = percent_changes_df.cov()
    cholesky = np.linalg.cholesky(covMatrix)
    
    meansMatrix = np.full(shape=(days,len(weights)),fill_value=means)
    meansMatrix = meansMatrix.T

    end = np.full(shape=(days + 1, num), fill_value= 0.0)
    end[0] = intiial_value

    for i in range(num):
        normal = np.random.normal(size = (days , len(weights)))
        returns = meansMatrix + np.inner(cholesky,normal) #inner product is inner product of each row cholesky to each row of normal is equal to cholesky @ normal.T
        end[1:,i] = np.cumprod(np.inner(weights, returns.T) + 1) * intiial_value

    return pd.DataFrame(end)
   

def Graphing_results(results:dict):
    for ticker in results:
        sns.histplot(results[ticker], kde= True, bins='auto', stat='density')
        plt.title(f"{ticker} Results")
        plt.xlabel("Price")
        plt.show()

def df_log_changes(data: dict):
    end = pd.DataFrame(index=data[list(data.keys())[0]].index)
    for ticker in data:
        end[ticker] = data[ticker]['Log Change']
    return end

def avg_returns(log_changes: pd.DataFrame):
    '''Gives YEARLY historical expected returns. Given by daily expected * 252'''
    end = dict()
    for ticker in log_changes.columns.tolist():
        end[ticker] = np.mean(log_changes[ticker]) * 252
    return end

def random_weights(assets:int, num = 10000):
    end = np.random.uniform(0,1,size=(num,assets))
    sums = np.sum(end,axis=1)
    end = end / sums[:,np.newaxis] #scale for sum of each to be equal to 1
    return end

def port_exp_vol(weights: np.ndarray, expected_array : np.ndarray, covariance: np.ndarray):
    expected = np.sum(weights * expected_array, axis= 1) #sum weight * expected_hist_return
    vols = []
    for i in range(len(weights)):
        vols.append(weights[i].T @ covariance @ weights[i])
    return expected, np.sqrt(np.array(vols))

def plot_frontier(expected_returns: np.ndarray, volatilities: np.ndarray, r = 4.11/100):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, expected_returns, c = (expected_returns - r) / volatilities)
    plt.title('Portfolio Frontier')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Expected Return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.grid(True)
    plt.show()
    
def Sharpe_ratio_negative(weights, expected, covariance, r = 4.11/100):
    expected = np.sum(weights * expected) #sum weight * expected_hist_return
    vol = np.sqrt(weights.T @ covariance @ weights)
    ratio = (expected - r) / vol
    return -ratio

def max_min_sharpe(weights, expected, vol, r = 4.11/100):
    sharpe = (expected - r) / vol
    max_sharpe = np.argmax(sharpe)
    max_weights = weights[max_sharpe]
    min_sharpe = np.argmin(sharpe)
    min_weighs = weights[min_sharpe]
    return max_weights, min_weighs


def sum_one(weights):
    return np.sum(weights) - 1

port = ['NVDA','MSFT','JNJ']
data, closes = get_data(port)
log_changes = df_log_changes(data)
#results = Port_GBM(data)
covariance = log_changes.cov()
year_cov = covariance * 252
cholesky_decom = np.linalg.cholesky(year_cov)
year_expected = avg_returns(log_changes)
year_expected_array = [year_expected[ticker] for ticker in year_expected]
dailey_expected = [i/252 for i in year_expected_array]
weights = random_weights(len(port))


expected_return, vol = port_exp_vol(weights,year_expected_array,year_cov)
plot_frontier(expected_return,vol)
max_sharpe, min_sharpe = max_min_sharpe(weights,expected_return,vol)


best = minimize(Sharpe_ratio_negative,np.array([1/len(port)] * len(port)),args= (year_expected_array,year_cov,4.11/100),method='SLSQP',bounds=((0,1),)*len(port), constraints= ({'type':'eq','fun':sum_one}))



print(f'Monte Carlo Max Sharpe Ratio weights:')
for i in range(len(port)):
    print(f'{port[i]}: {round(100 * max_sharpe[i],2)}')
print(f'with a ratio of {-Sharpe_ratio_negative(max_sharpe,year_expected_array,year_cov)}')
print('\n')

print(f'Monte Carlo Min Sharpe Ratio weights:')
for i in range(len(port)):
    print(f'{port[i]}: {round(100 * min_sharpe[i],2)}')
print(f'with a ratio of {-Sharpe_ratio_negative(min_sharpe,year_expected_array,year_cov)}')

print('\n')
print(f'Minimize Max Sharpe Ratio weights:')
for i in range(len(port)):
    print(f'{port[i]}: {round(100 * best.x[i],2)}')
print(f'with a ratio of {-Sharpe_ratio_negative(best.x,year_expected_array,year_cov)}')
print('\n')

port_sim = Port_GBM(data,best.x)

port_sim.plot(legend=False)
plt.title("Future Price")
plt.xlabel("Days")
plt.show()


sns.histplot(np.array(np.array(port_sim.iloc[-1])), kde= True, bins='auto', stat='density')
plt.title("Results")
plt.xlabel("Price")
plt.show()