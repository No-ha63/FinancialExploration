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

def ProbabilityOver(results,last_close, mu, sigma):
    np.array(results)
    last_close = round(last_close,2)
    print(f"Probabilty over last close ({last_close}) by count: {np.count_nonzero(results > last_close)/len(results)}")
    print(f"Probabilty over last close ({last_close}) by Log Normal: {round(1 - norm.cdf(m.log(last_close), mu, sigma),4)}") #approximation


def ROI(last_close,mean, mu, sigma):
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

        value = m.e ** norm.ppf(.75,mu,sigma)
        print('\n')
        print(f"For a little more risk, buy {m.ceil(target/(value-last_close))} shares at {round(m.ceil(target/(value-last_close)) * last_close,2)} for a 25% chance your target will be met or exceeded.")
    else:
        print(f"Expected return off of expected value ({mean}) at last close ({last_close}): {expected_return}")
        print("Hence, investing into this stock is very risky!!")

def convergence(data,jumps):
    #testing for convergence on expected value
    many_df = pd.DataFrame(columns=["Runs","Expected"])
    runs = []
    expected = []
    x = 1000
    for i in range(x,10000+1,x):
        runs.append(i)
        expected.append(np.mean(GBM_Jump(data, i, 252).iloc[-1],jumps))

    many_df["Runs"] = runs
    many_df["Expected"] = expected
    print(many_df)


def get_jumps(log_changes):
    return [i for i in log_changes if abs(i - np.mean(log_changes)) > 3.5 * np.std(log_changes)]

def GBM_Jump(data,num,days, window = total):
    #There is the possibilty of 0 jumps in a certain window / if only one jump in period, all jumps will be of that size in model
    log_change = data["Log Change"][- window : ]
    jumps = get_jumps(log_change)
    print(f"There were {len(jumps)} jump(s) in the last {window} days.")

    mu = np.mean(log_change)
    std = np.std(log_change)
    

    if len(jumps) > 0:
        jump_mu = np.mean(jumps)
        jump_std = np.std(jumps)
    
        
        k = (m.e ** (jump_mu + (1/2) * jump_std ** 2)) - 1
        lam = len(jumps) / window

        drift = mu - (1/2) * (std ** 2) - lam * k

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
    elif len(jumps) == 0: #normal GBM if no jumps in given window
        drift = mu - (1/2) * (std ** 2)
        z_updated = np.exp(np.random.normal(loc = drift, scale = std, size = (days,num))) 
        end = np.zeros(shape=(days+1,num))
        end[0] = data['Close'][-1]
        for i in range(1,days+1):
            end[i] = end[i-1] * z_updated[i-1] #Multiplying lognormal by lognormal produces lognormal
    
        return pd.DataFrame(end)


ticker = input("What stock would you like to predict ->")
answer = 0
trade_days = 252
while answer <= 0 or answer > 3:
    answer = int(input("Would you like to model for (1) Days, (2) Weeks, (3) A Year ->"))
    if answer == 1:
        days = int(input("How many days ->"))
        trade_days = days
    elif answer == 2:
        weeks = int(input("How many weeks ->"))
        trade_days = 5 * weeks
    elif answer == 3:
        trade_days = 252


print('\n'*2)
stock = yf.Ticker(ticker)
date_start = dt.date.today() - dt.timedelta(days = total)
data = stock.history(start = date_start)
data = data.drop(columns=['Volume','Dividends','Stock Splits'])

data["Close"].plot(figsize= (12,5))
plt.title(f"{ticker} Stock Prices")
plt.show()


data["Percent Change"] = data["Close"].pct_change()
data["Log Change"] = np.log(data['Percent Change'] + 1)

sns.histplot(data["Log Change"], kde= True, bins='auto', stat='density')
plt.title("Log of Percant Changes")
plt.xlabel("Price")
plt.show()

print("Predictions with Jumps")


n = 10000
sim_df = GBM_Jump(data, n, trade_days)
results = np.array(sim_df.iloc[-1]) #lognormal dist
expected = np.mean(results)
spread = np.std(results, ddof=1) #1/(n-1) sum(from i = 1 to n) (xi - Xbar)^2. which is used in the approximate pivot for CLT
clt_quant = 1.76

#MLE for mu is (1/n)sum(ln(xi))
#MLE for mu is (1/n)sum((ln(xi)-mu)^2)
#MLE's are asymptocally consistent and fit Cramer Rao Lower Bound
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

ROI(data["Close"][-1],expected,mle_mu,mle_sigma)

print('\n')

