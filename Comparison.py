import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import yfinance as yf
import math as m
import scipy.stats as st


def get_jumps(log_changes):
    return [i for i in log_changes if abs(i - np.mean(log_changes)) > 3.5 * np.std(log_changes)]

def GBM(data,num,days):
    log_change = data["Log Change"]
    mu = np.mean(log_change)
    var = np.var(log_change)
    std = np.std(log_change)
    drift = mu - (1/2) * var

    #z = np.random.normal(size=(days,num)) 

    # X = mu + sigma * Z where Z is N(0,1) gives N(mu,sigma^2)
    #delta_t = 1 as using daily increments and data for drift and volatility are daily closings
    z_updated = np.exp(np.random.normal(loc = drift, scale = std, size = (days,num))) 
    end = np.zeros(shape=(days+1,num))
    end[0] = data['Close'][-1]
    for i in range(1,days+1):
        end[i] = end[i-1] * z_updated[i-1] #Multiplying lognormal by lognormal produces lognormal
    
    #//To run as n^2 iterative
    #   for i in range(num):
    #     running = [data['Close'][-1]]
    #     for j in range(days):
    #         z = np.random.normal()
    #         running.append(running[-1] * (exp ** (drift + std * z))) #makes a lognormal dist
    #     end.append(running)
    return pd.DataFrame(end)


def GBM_Jump(data,num,days,jumps,total_past):
    log_change = data["Log Change"]
    mu = np.mean(log_change)
    var = np.var(log_change)
    std = np.std(log_change)
    
    jump_mu = np.mean(jumps)
    jump_std = np.std(jumps)
    jump_var = np.var(jumps)
    k = m.e ** (jump_mu + (1/2) * jump_var) - 1
    lam = len(jumps) / total_past

    drift = mu - (1/2) * var - lam * k

    # X = mu + sigma * Z where Z is N(0,1) gives N(mu,sigma^2)
    #delta_t = 1 as using daily increments and data for drift and volatility are daily closings
    z = np.random.normal(loc = drift, scale = std, size = (days,num))
    num_jumps = np.random.poisson(lam = lam , size= (days,num))
    jump_size = np.random.normal(loc = jump_mu, scale = jump_std, size=(days,num))
    jump_for_day = num_jumps * jump_size

    #to complete iteratively
    # for i in range(days):
    #     for j in range(num):
    #         z[i][j] = z[i][j] + np.sum(np.random.normal(loc = jump_mu, scale = jump_std, size = num_jumps[i][j]))
    
    z_updated = np.exp(z + jump_for_day)
    end = np.zeros(shape=(days+1,num))
    end[0] = data['Close'][-1]
    for i in range(1,days+1):
        end[i] = end[i-1] * z_updated[i-1] #Multiplying lognormal by lognormal produces lognormal
    
    return pd.DataFrame(end)

print('\n' * 2)
ticker = input("What stock would you like to predict ->")
answer = 0
total = 3 * 365
trade_days = 252


print('\n'*2)
stock = yf.Ticker(ticker)
date_start = dt.date.today() - dt.timedelta(days = total)
data = stock.history(start = date_start)
data = data.drop(columns=['Volume','Dividends','Stock Splits'])



data["Percent Change"] = data["Close"].pct_change()
data["Log Change"] = np.log(data['Percent Change'] + 1)

jumps = get_jumps(data['Log Change'])

n = 10000
jump_sim_df = GBM_Jump(data, n, trade_days,jumps,total)
jump_results = np.array(jump_sim_df.iloc[-1]) #lognormal dist

sim_df = GBM(data, n, trade_days)
results = np.array(sim_df.iloc[-1]) #lognormal dist

expected = np.mean(results)
jump_expected = np.mean(jump_results)
spread = np.std(results)
jump_spread = np.std(jump_results)


print(f"Price at last close: {round(data['Close'][-1],2)}")

print("Statistics for GBM:")
print(f"Expected: {round(expected,2)} Deviation: {round(spread,2)}")
print(f"Median: {round(np.median(results),2)} ; Q1: {round(np.percentile(results,25),2)}  ; Q3 {round(np.percentile(results,75),2)} ; IQR: {round(np.percentile(results,75) - np.percentile(results,25),2)}")
print(f"90% Confidence Interval for Outcome: [{round(np.percentile(results,5),2)},{round(np.percentile(results,95),2)}]")
print(f"Kurtosis: {st.kurtosis(expected)}")

print("\nStatistics for Jump Diffusion")
print(f"Expected: {round(jump_expected,2)} Deviation: {round(jump_spread,2)}")
print(f"Median: {round(np.median(jump_results),2)} ; Q1: {round(np.percentile(jump_results,25),2)}  ; Q3 {round(np.percentile(jump_results,75),2)} ; IQR: {round(np.percentile(jump_results,75) - np.percentile(jump_results,25),2)}")
print(f"90% Confidence Interval for Outcome: [{round(np.percentile(jump_results,5),2)},{round(np.percentile(jump_results,95),2)}]")
print(f"Kurtosis: {st.kurtosis(jump_expected)}")

df = pd.concat([pd.DataFrame({"Prices" : results, "Type": ['GBM' for _ in range(len(results))]}), pd.DataFrame({"Prices" : jump_results, "Type": ['Jump' for _ in range(len(results))]})])
bxplt = sns.boxplot(data = df , x = 'Prices', y = 'Type', hue = 'Type', palette = ['b','orange'],legend = False)
plt.show()


sns.histplot(data= results, color = 'Red', alpha = .2, kde=True, label ='GBE', stat= 'density')
sns.histplot(data= jump_results, color = 'Green', alpha = .2, kde=True, label ='Jump GBE', stat= 'density')

plt.legend()
plt.show()