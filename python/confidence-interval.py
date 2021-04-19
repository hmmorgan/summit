
# 12 = number of samples
# mean = 14.5
# standard deviation = 4.3 
# for uniform distribution (if not use bootstrapping, credible interval)
stats.t.interval(1 - 0.2, 12 - 1, loc=14.5, scale= 4.3 / np.sqrt(12))

# another way?
alpha = 0.05                       # significance level = 5%
df = len(arr) - 1                  # degress of freedom = 20
t = stats.t.ppf(1 - alpha/2, df)   # t-critical value for 95% CI = 2.093
s = np.std(arr, ddof=1)            # sample standard deviation = 2.502
n = len(arr)

lower = np.mean(arr) - (t * s / np.sqrt(n))
upper = np.mean(arr) + (t * s / np.sqrt(n))