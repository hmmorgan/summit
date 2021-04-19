import utils as ut
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

# How noisy is the data on the GPU?

# Pick a machine
machine = "GPU" 
file_prefix = "vec_ops.n6_g1_c7_a1."
file_path = "noise"

# Pick an operation 
# operation = "VecDot"
# count = 1
operation = "VecAXPY"
count = 3

# Get the data
vec_size_ = [1e3, 1e4, 1e5, 1e6, 1e7, 1e9]
vec_size_string = ["10_3", "10_4", "10_5", "10_6", "10_7", "10_9"]

means = []
std_devs = []
conf_intervals = []
data = []
for size in vec_size_string:
	for i in range(1, 11):
		data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + size + "." + str(i), operation, count)))


	# error-bar = one standard deviation
	mean_ = scipy.mean(data)
	std_ = np.std(data)
	interval_ = stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data))
	# interval_ = stats.norm.interval(0.95, loc=mean_, scale=std_)
	print interval_

	means.append(mean_)
	std_devs.append(std_)
	conf_intervals.append(interval_)


	data = []

print conf_intervals

	# error-bar = 95% confidence interval


file_path = "waitforgpu-latency"

# Get the data
sm = range(1000, 100001, 100)
md = range(100000, 10000001, 10000) 
lg = range(10000000, 100000001, 100000)
vec_sizes = sm + md + lg

all_data = []
for size in vec_sizes:
	all_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))

# Compute the moving average
span_size = 10
internal_sizes = vec_sizes[span_size: -span_size]
moving_average = []

for vec_size in internal_sizes: # Loop through internal data
	ind = vec_sizes.index(vec_size) # Get place in global vector
	moving_data = all_data[ind - span_size: ind + span_size] # Get local data from global data
	avg = sum(moving_data)/len(moving_data) # Compute average of local data
	moving_average.append(avg)

# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111)

# for i in range(len(vec_size)):
# 	plt.plot([vec_size[i]]*10, data[i], linestyle="none", marker=".", color="grey")

data_df=np.array([10]*6)

# plt.errorbar(vec_size_, means, yerr=conf_intervals, linestyle="none", color="blue", linewidth=2, label="95% confidence")
# plt.errorbar(vec_size_, means, yerr=stats.t.ppf(0.95, data_df)*std_devs, linestyle="none", color="blue", linewidth=2, label="95% confidence")
plt.errorbar(vec_size_, means, yerr=std_devs, linestyle="none", color="red", linewidth=2, label="One standard deviation")

plt.plot(vec_sizes, all_data, linestyle="none", marker=".", color="grey", label="Data")
plt.plot(internal_sizes, moving_average, color="black", label="Moving average")
plt.title(machine + " " + operation + " execution time", fontsize=12)
plt.xlabel("Vector size", fontsize=12)
plt.ylabel("Seconds", fontsize=12)
plt.legend(loc="upper left", fontsize=12, frameon=False)
plt.xscale('log')
plt.xlim([8e2, 1.2e7])
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.xaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
plt.show()
