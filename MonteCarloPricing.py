import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import yfinance as yf
import math as m
from scipy.stats import norm

month_1_rate = 4.06/100
compound_rate = np.log(1 + month_1_rate)
total = 1 * 365

def GBM_risk_neutral(data,implied_volatility, num = 1000000, r = compound_rate, years = 1.0):
    std = implied_volatility #annual
    drift = r - (1/2) * std**2 #annual

    # X = mu + sigma * Z where Z is N(0,1) gives N(mu,sigma^2)
    # z_updated = np.exp(np.random.normal(loc = drift * Dt, scale = std * np.sqrt(Dt), size = (days,num))) 
    # end = np.zeros(shape=(days+1,num))
    # end[0] = data['Close'][-1]
    # for i in range(1,days+1):
    #     end[i] = end[i-1] * z_updated[i-1] #Multiplying lognormal by lognormal produces lognormal

    #above is to do iterativly, can do it from just results as GBM as solution

    results =  data['Close'][-1] * np.exp(np.random.normal(loc= drift * years, scale= std * np.sqrt(years), size = num))
    return results

def pricing(results, K, r = compound_rate, years = 1.0, mode = 'call'):
    if mode == 'call':
        all_end = np.maximum(results - K, 0)
        call_payoff = np.mean(all_end)
        return np.exp(-r * years) * call_payoff
    elif mode == 'put':
        all_end = np.maximum(K - results, 0)
        put_payoff = np.mean(all_end)
        return np.exp(-r * years) * put_payoff
    else:
        print("Mode must be 'call' or 'put'")
    
def black_scholes(last_close, sigma, K, r = compound_rate, years = 1.0, mode = 'call'):
    if mode == 'call':
        d1 = (np.log(last_close/K) + (r + (1/2) * sigma **2) * years) / sigma * np.sqrt(years) 
        d2 = d1 - sigma * np.sqrt(years)
        call_price = last_close * norm.cdf(d1) - norm.cdf(d2) * K * np.exp(-r * years)
        return call_price
    else:
        d1 = (np.log(last_close/K) + (r + (1/2) * sigma **2) * years) / sigma * np.sqrt(years)
        d2 = d1 - sigma * np.sqrt(years)
        put_price = norm.cdf(-d2) * K * np.exp(-r * years) - norm.cdf(-d1) * last_close
    return put_price

ticker = input("What stock would you like to predict ->").upper()
stock = yf.Ticker(ticker)
date_start = dt.date.today() - dt.timedelta(days = total)
data = stock.history(start = date_start)
data = data.drop(columns=['Volume','Dividends','Stock Splits'])

data["Percent Change"] = data["Close"].pct_change() 
data["Log Change"] = np.log(data['Percent Change'] + 1) #normal dist

target = dt.date.today() + dt.timedelta(days= 30)
closest_date = stock.options[np.argmin(np.abs(np.array(stock.options,dtype='datetime64[D]').astype(dt.datetime) - target))]
options_data = stock.option_chain(closest_date) #options from the earliest expiration
# options_data = stock.option_chain(stock.options[0]) #closes day
# date = stock.options[0]


calls = options_data.calls.copy()
puts = options_data.puts.copy()


last_close = data['Close'][-1]

closest_call = calls['strike'].iloc[np.argmin(np.abs(calls['strike'] - last_close))]
closest_call_implied = calls['impliedVolatility'].iloc[np.argmin(np.abs(calls['strike'] - last_close))]

closest_put = puts['strike'].iloc[np.argmin(np.abs(puts['strike'] - last_close))]
closest_put_implied = puts['impliedVolatility'].iloc[np.argmin(np.abs(puts['strike'] - last_close))]

trade_years = (np.asarray(closest_date,dtype='datetime64[D]').astype(dt.datetime) - data.index[-1].date()) / dt.timedelta(365)


sim_call_implied = GBM_risk_neutral(data, years = trade_years, implied_volatility = closest_call_implied)
sim_put_implied = GBM_risk_neutral(data, years= trade_years, implied_volatility = closest_put_implied)

MC_call_price = pricing(sim_call_implied, closest_call, compound_rate, trade_years, mode= 'call')
MC_put_price = pricing(sim_put_implied, closest_put, compound_rate, trade_years, mode= 'put')
BS_call_price = black_scholes(last_close,closest_call_implied, closest_call,compound_rate,trade_years,'call')
BS_put_price = black_scholes(last_close,closest_put_implied, closest_put,compound_rate,trade_years,'put')

print(f"Call prices for {ticker} on {closest_date} at rate {month_1_rate * 100}, and strike {closest_call}:")
print(f"Monte Carlo: {round(MC_call_price,2)}")
print(f'Black-Scholes: {round(BS_call_price,2)} ')

print()
print(f"Put prices for {ticker} on {closest_date} at rate {month_1_rate * 100} and strike {closest_put}:")
print(f"Monte Carlo: {round(MC_put_price,2)}")
print(f'Black-Scholes: {round(BS_put_price,2)}')

