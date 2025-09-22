import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import yfinance as yf
import math as m
from scipy.stats import norm


print('\n'*2)

total = 3 * 365

def GBM(data,num = 10000, days = 252, window = total):
    log_change = data["Log Change"][- window : ]
    mu = np.mean(log_change)
    std = np.std(log_change)
    drift = mu - (1/2) * (std**2)
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

def ProbabilityOver(results,last_close, mu, sigma):
    np.array(results)
    last_close = round(last_close,2)
    print(f"Probabilty over last close ({last_close}) by count: {np.count_nonzero(results > last_close)/len(results)}")
    print(f"Probabilty over last close ({last_close}) by Log Normal: {round(1 - norm.cdf(m.log(last_close), mu, sigma),4)}") #approximation


def ROI(last_close,results):
    mean = np.mean(results)
    expected_return = round(mean - last_close,2)
    last_close = round(last_close,2)
    mean = round(mean,2)
    print("\n")
    if expected_return > 0:
        print(f"Expected return off of expected value ({mean}) at last close ({last_close}): {expected_return}")
        print(f"In percent: {round((mean - last_close)/last_close * 100,2)}")
        print("\n")


        target = float(input("What is your target to make ->"))
        print(f"With a target of ${target}, {m.ceil(target/expected_return)} shares are needed to be purchased, for a purchase of: {round(m.ceil(target/expected_return) * last_close,2)} at last close.")
        print("With the expected price outcome, this will ensure the goal is met")

        value = np.percentile(results,75)
        print('\n')
        print(f"For a little more risk, buy {m.ceil(target/(value-last_close))} shares at {round(m.ceil(target/(value-last_close)) * last_close,2)} for a 25% chance your target will be met or exceeded.")
    else:
        print(f"Expected return off of expected value ({mean}) at last close ({last_close}): {expected_return}")
        print("Hence, investing into this stock is very risky!!")

def convergence(data):
    #testing for convergence on expected value
    many_df = pd.DataFrame(columns=["Runs","Expected"])
    runs = []
    expected = []
    x = 1000
    for i in range(x,10000+1,x):
        runs.append(i)
        expected.append(np.mean(GBM(data, i, 252).iloc[-1]))

    many_df["Runs"] = runs
    many_df["Expected"] = expected
    print(many_df)

def VaR_CVaR(results: pd.Series, mu: float, sigma: float, last_close: float):
    value_90 = round(np.percentile(results,10),2)
    value_95 = round(np.percentile(results, 5),2)
    value_99 = round(np.percentile(results, 1),2)

    value_90_log = round(m.e ** norm.ppf(.10,mu,sigma),2)
    value_95_log = round(m.e ** norm.ppf(.05,mu,sigma),2)
    value_99_log = round(m.e ** norm.ppf(.01,mu,sigma),2)

    mean_90 = round(np.mean([x for x in results if x <= value_90]),2)
    mean_95 = round(np.mean([x for x in results if x <= value_95]),2)
    mean_99 = round(np.mean([x for x in results if x <= value_99]),2)

    print('\n')
    print(f'Value at which 10% of predictions fall under by results: {value_90}')
    print(f'Value at which 10% of predictions fall under by lognormal approximation: {value_90_log}')
    print(f'Expected price of lower 10%: {mean_90}')
    if mean_90 - last_close < 0:
        print(f'Expected loss: {round(abs(mean_90 - last_close),2)}') #E[X - last_close] = E[X] - last_close
    else:
        print(f'Expected gain: {round(abs(mean_90 - last_close),2)}')
    
    print('\n')
    print(f'Value at which 5% of predictions fall under by results: {value_95}')
    print(f'Value at which 5% of predictions fall under by lognormal approximation: {value_95_log}')
    print(f'Expected price of lower 5%: {mean_95}')
    if mean_95 - last_close < 0:
        print(f'Expected loss: {round(abs(mean_95 - last_close),2)}') #E[X - last_close] = E[X] - last_close
    else:
        print(f'Expected gain: {round(abs(mean_95 - last_close),2)}')

    print('\n')
    print(f'Value at which 1% of predictions fall under by results: {value_99}')
    print(f'Value at which 1% of predictions fall under by lognormal approximation: {value_99_log}')
    print(f'Expected price of lower 1%: {mean_99}')
    if mean_99 - last_close < 0:
        print(f'Expected loss: {round(abs(mean_99 - last_close),2)}') #E[X - last_close] = E[X] - last_close
    else:
        print(f'Expected gain: {round(abs(mean_99 - last_close),2)}')



ticker = input("What stock would you like to predict ->").upper()
answer = float(input("How many years ->"))
trade_days = m.ceil(answer * 252)

print('\n'*2)
stock = yf.Ticker(ticker)
date_start = dt.date.today() - dt.timedelta(days = total)
data = stock.history(start = date_start)
data = data.drop(columns=['Volume','Dividends','Stock Splits'])

data["Close"].plot(figsize= (12,5))
plt.title(f"{ticker} Stock Prices")
plt.show()


data["Percent Change"] = data["Close"].pct_change() 
data["Log Change"] = np.log(data['Percent Change'] + 1) #normal dist

sns.histplot(data["Percent Change"], kde= True, bins='auto', stat='density')
plt.title("Percant Changes")
plt.xlabel("Price")
plt.show()

sns.histplot(data["Log Change"], kde= True, bins='auto', stat='density')
plt.title("Log of Percant Changes")
plt.xlabel("Price")
plt.show()

n = 10000
sim_df = GBM(data, n, trade_days)
results = np.array(sim_df.iloc[-1]) #lognormal dist
expected = np.mean(results)
spread = np.std(results, ddof=1) #1/(n-1) sum(from i = 1 to n) (xi - Xbar)^2. which is used in the approximate pivot for CLT
clt_quant = 1.76

#MLE for mu is (1/n)sum(ln(xi))
#MLE for sigma is (1/n)sum((ln(xi)-mu)^2)
#MLE's are asymptocilly consistent and fit Cramer Rao Lower Bound
mle_mu = np.mean(np.log(results))
mle_sigma = np.std(np.log(results)) 

sim_df.plot(legend=False)
plt.title("Future Price")
plt.xlabel("Days")
plt.show()


sns.histplot(np.log(results), kde= True, bins='auto', stat='density')
plt.title("Log of Results")
plt.xlabel("Price")
plt.show()

sns.histplot(results, kde= True, bins='auto', stat='density')
plt.title("Results")
plt.xlabel("Price")
plt.show()


print(f"Price at last close: {round(data['Close'][-1],2)}")
print(f"Expected: {round(expected,2)} Deviation: {round(spread,2)}")
print(f"Median: {round(np.median(results),2)} ; Q1: {round(np.percentile(results,25),2)}  ; Q3 {round(np.percentile(results,75),2)} ; IQR: {round(np.percentile(results,75) - np.percentile(results,25),2)}")
print("\n"*2)
print(f"90% Confidence Interval for Outcome: [{round(np.percentile(results,5),2)},{round(np.percentile(results,95),2)}]")

clt_upper = round(expected + (spread/(n**(1/2))) * clt_quant , 2) # using (Xbar - mu) / (S/sqrt(n) as an approximate pivot for confidence intervals 
clt_lower = round(expected - (spread/(n**(1/2))) * clt_quant , 2)
print(f"95% Confidence Interval for Real Mean: [{clt_lower},{clt_upper}]")

ProbabilityOver(results,data["Close"][-1],mle_mu,mle_sigma)

ROI(data["Close"][-1],results)

VaR_CVaR(results,mle_mu,mle_sigma,data["Close"][-1])

print('\n')

date_start = dt.date.today() - dt.timedelta(days = 365 + total)
new_data = stock.history(start = date_start)
new_data = new_data.drop(columns=['Volume','Dividends','Stock Splits'])

cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days = 364), tz= 'UTC') #exclusive end of slice
new_data = new_data.loc[:cutoff]
new_data["Percent Change"] = new_data["Close"].pct_change()
new_data["Log Change"] = np.log(new_data['Percent Change'] + 1)

new_data["Close"].plot(figsize= (12,5))
plt.title(f"{ticker} Past Stock Prices")
plt.show()

inclusive_cutoff = new_data.index[-1]
test_sim_df = GBM(new_data, n, len(data.loc[inclusive_cutoff:]['Close'])-1)
test_results = np.array(test_sim_df.iloc[-1]) #lognormal dist
test_expected = np.mean(test_results)
test_spread = np.std(test_results)

print("Testing Model by esitmating price of today from a year ago")

print(f"Price at last close: {round(new_data['Close'][-1],2)}")
print(f"Expected: {round(test_expected,2)} Deviation: {round(test_spread,2)}")
print(f"Median: {round(np.median(test_results),2)} ; Q1: {round(np.percentile(test_results,25),2)}  ; Q3 {round(np.percentile(test_results,75),2)} ; IQR: {round(np.percentile(results,75) - np.percentile(results,25),2)}")
print(f"90% Confidence Interval for Outcome: [{round(np.percentile(test_results,5),2)},{round(np.percentile(test_results,95),2)}]")
print(f"Actual Price today: {round(data['Close'][-1],2)}")
print('\n'*2)


means = []
upper = []
lower = []

for i in range(len(test_sim_df.index)):
    means.append(np.mean(test_sim_df.iloc[i]))
    upper.append(np.percentile(test_sim_df.iloc[i],95))
    lower.append(np.percentile(test_sim_df.iloc[i],5))


fig, ax = plt.subplots()
ax.plot(range(len(means)),means, label = 'Expected', color = 'black')
ax.plot(range(len(means)),data.loc[inclusive_cutoff:]["Close"],label = 'Actual', color = 'red')
ax.fill_between(x = range(len(means)), y1=upper, y2=lower, alpha = .4, label = '90% Range')
ax.fill_between(range(len(means)), upper, data.loc[inclusive_cutoff:]["Close"], where= data.loc[inclusive_cutoff:]["Close"] > upper, fc='orange', alpha=0.4)
ax.fill_between(range(len(means)), lower, data.loc[inclusive_cutoff:]["Close"], where= data.loc[inclusive_cutoff:]["Close"] < lower, fc='orange', alpha=0.4)
ax.set_title("Model VS Actual")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend(loc = 'upper left')
plt.show()