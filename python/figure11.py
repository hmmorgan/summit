import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy

# Average for GPU data
def get_gpu_data(sizes, operation, count):

	file_prefix = "vec_ops.n6_g1_c7_a1"
	file_path = "gpu-statistics"
	num_samples = 10

	all_data = []
	for size in sizes:
		one_vector_size = []
		for sample in range(1, num_samples + 1):
			file = "../data/{}/{}.{}.{}".format(file_path, file_prefix, size, sample)
			one_vector_size.append(float(1000000*ut.get_time(file, operation, count)))
		all_data.append(one_vector_size)

	average = []
	for i in range(len(sizes)):
		one_vector_size = all_data[i]
		average.append(sum(one_vector_size)/num_samples)

	return all_data, average

# Plot GPU data and average
def plot_gpu_data_and_average(sizes, operation, count):

	all_data, average = get_gpu_data(sizes, operation, count)

	# Make the plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in range(len(sizes)):
		if i == 0:
			plt.plot([sizes[i]]*10, all_data[i], linestyle="none", marker=".", color="grey", label="Data")
		else:
			plt.plot([sizes[i]]*10, all_data[i], linestyle="none", marker=".", color="grey")
	plt.plot(sizes, average, color="black", label="Mean")
	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper right", fontsize=12, frameon=False)
	plt.xlim([8e2, 1.2e5])
	ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

	plt.tight_layout()
	plt.savefig("../plots/" + operation + "_GPU_mean.png")
	plt.show()

def fig11_cpu(sizes, operation1, count1, operation2, count2, show):

	cpu_time1 = []
	cpu_time2 = []

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation1, count1)
		cpu_time1.append(1000000*time) # cheating so that y-axis units are 10^-6

		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation2, count2)
		cpu_time2.append(1000000*time)

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	# plt.plot(sizes, cpu_time1, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs " + operation1)
	plt.plot(sizes, cpu_time2, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs " + operation2)
	plt.title("CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="lower right", fontsize=12, frameon=False)
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/" + operation2 + "CPU_sm.png")
	# plt.savefig("../plots/CPU_" + operation1 + "_" + operation2 + "_time_42CPU.png")
	if show: plt.show()

def fig11_gpu(operation1, count1, operation2, count2, show, log=False):

	_, average1 = get_gpu_data(sizes, operation1, count1)
	_, average2 = get_gpu_data(sizes, operation2, count2)

	gpu_time1 = []
	gpu_time2 = []

	for time in average1:
		gpu_time1.append(1000000*time) # cheating so that y-axis units are 10^-6
	for time in average2:
		gpu_time2.append(1000000*time)

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	plt.plot(sizes, gpu_time1, marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation1)
	plt.plot(sizes, gpu_time2, marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation2)
	plt.title("Mean GPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="center left", fontsize=12, frameon=False)
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
	if log: plt.xscale('log')
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/GPU_" + operation1 + "_" + operation2 + "_time_6GPU.png")
	if show: plt.show()

def gpu_error_bars(sizes, operation, count, log=False):

	all_data, average = get_gpu_data(sizes, operation, count)

	# Error bars
	means = []
	height = []
	above = []
	below = []

	def mean_confidence_interval(data, confidence=0.95):
	    a = 1.0 * np.array(data)
	    n = len(a)
	    # m, se = np.mean(a), scipy.stats.sem(a)
	    m, s = np.mean(a), np.std(a)
	    t = scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	    h = t*s/np.sqrt(n)
	    return m, m-h, m+h, h	

	for one_vector_size in all_data:

		# temp_mean = scipy.mean(one_vector_size)
		# temp_std = np.std(one_vector_size)

		# means.append(temp_mean)
		# std_devs.append(temp_std)

		# above.append(temp_mean + temp_std)
		# below.append(temp_mean - temp_std)

		m, b, a, h = mean_confidence_interval(one_vector_size)
		means.append(m)
		below.append(b)
		above.append(a)
		height.append(h)

	# Make the plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])
	plt.errorbar(sizes, means, yerr=height, linestyle="none", linewidth=2, label="95% confidence")
	# plt.plot(sizes, above, color="black", label="95% confidence")
	# plt.plot(sizes, below, color="black")
	plt.plot(sizes, average, label="Mean")
	

	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper center", fontsize=12, frameon=False)
	if log: plt.xscale('log')
	ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

	plt.tight_layout()
	plt.savefig("../plots/" + operation + "_GPU_error_bar_95.png")
	plt.show()

def gpu_error_bars_with_data(sizes, operation, count, log=False):

	all_data, average = get_gpu_data(sizes, operation, count)
	all_sizes = sizes

	# Calculate error bars
	means = []
	height = []
	above = []
	below = []
	std_devs = []

	def mean_confidence_interval(data, confidence=0.95):
	    a = 1.0 * np.array(data)
	    n = len(a)
	    m, s = np.mean(a), np.std(a)
	    t = scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	    h = t*s/np.sqrt(n)
	    return m, s, h

	for one_vector_size in all_data:

		m, s, h = mean_confidence_interval(one_vector_size)
		means.append(m)
		std_devs.append(s)

		# Error bar: 95% confidence interval
		below.append(m-h)
		above.append(m+h)
		height.append(h)

		# Error bar: one standard deviation
		# below.append(m-s)
		# above.append(m+s)

	# Make the plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])
	for i in range(len(all_sizes)):
		if i == 0:
			plt.plot([all_sizes[i]]*10, all_data[i], linestyle="none", marker=".", markeredgecolor="None", color="grey", alpha=0.4, label="Data")
		else:
			plt.plot([all_sizes[i]]*10, all_data[i], linestyle="none", marker=".", markeredgecolor="None", color="grey", alpha=0.4)
	# plt.errorbar(sizes, means, yerr=height, linestyle="none", linewidth=2, label="95% confidence")
	plt.plot(sizes, above, label="95% confidence")
	plt.plot(sizes, below, color="black")
	plt.plot(sizes, means, label="Mean")
	

	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	if log: plt.xscale('log')
	# plt.xlim([-100, 1001000])
	ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

	plt.tight_layout()
	# plt.savefig("../plots/" + operation + "_GPU_mean_with_data_sm.png")
	# plt.savefig("../plots/" + operation + "_GPU_error_bar_with_data_md_.png")
	plt.show()


def gpu_error_bars_with_data_moving_average(sizes, operation, count, log=False):

	all_data, average = get_gpu_data(sizes, operation, count)
	all_sizes = sizes

	# Calculate error bars
	means = []
	height = []
	above = []
	below = []
	std_devs = []

	def mean_confidence_interval(data, confidence=0.95):
	    a = 1.0 * np.array(data)
	    n = len(a)
	    m, s = np.mean(a), np.std(a)
	    t = scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	    h = t*s/np.sqrt(n)
	    return m, s, h

	for one_vector_size in all_data:

		m, s, h = mean_confidence_interval(one_vector_size)
		means.append(m)
		std_devs.append(s)

		# Error bar: 95% confidence interval
		below.append(m-h)
		above.append(m+h)
		height.append(h)

		# Error bar: one standard deviation
		# below.append(m-s)
		# above.append(m+s)

	# Compute the moving average
	span_size = 10
	internal_sizes = sizes[span_size: -span_size]
	moving_means = []
	moving_heights = []
	moving_above = []
	moving_below = []

	for vec_size in internal_sizes: # Loop through internal data
		ind = sizes.index(vec_size) # Get place in global vector
		moving_data = means[ind - span_size: ind + span_size] # Get local data from global data
		avg = sum(moving_data)/len(moving_data) # Compute average of local data
		moving_means.append(avg)

		moving_data = height[ind - span_size: ind + span_size]
		hgt = sum(moving_data)/len(moving_data) # Compute average of local data
		moving_heights.append(hgt)
		moving_above.append(avg+hgt)
		moving_below.append(avg-hgt)


	# Make the plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])
	# for i in range(len(all_sizes)):
	# 	if i == 2:
	# 		plt.plot([all_sizes[i]]*10, all_data[i], linestyle="none", marker=".", markeredgecolor="None", color="grey", alpha=1, label="Data")
	# 	elif all_sizes[i] == 10000000:
	# 		plt.plot([all_sizes[i]]*10, all_data[i], linestyle="none", marker=".", markeredgecolor="None", markersize=10, color="black", alpha=1)
	# 	else:
	# 		plt.plot([all_sizes[i]]*10, all_data[i], linestyle="none", marker=".", markeredgecolor="None", color="grey", alpha=1)
	# plt.errorbar(sizes, moving_means, yerr=moving_heights, linestyle="none", linewidth=2, label="95% confidence")
	plt.plot(internal_sizes, moving_above, label="95% confidence")
	plt.plot(internal_sizes, moving_below, color="black")
	plt.plot(internal_sizes, moving_means, label="Mean")
	

	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	if log: plt.xscale('log')
	# plt.ylim([85, 110]) # VecDot
	# plt.ylim([55, 85]) # VecAXPY
	ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

	plt.tight_layout()
	plt.savefig("../plots/" + operation + "_GPU_moving_average_95_lg.png")
	plt.show()


# All data
sm = range(1000, 100001, 1000)
md = range(100000, 10000001, 100000) 
lg = range(10000000, 100000001, 1000000)
sizes = sm + md + lg
# [1000, 10000, 100000, 1000000]

# plot_gpu_data_and_average(sizes, "VecDot", 1)
# plot_gpu_data_and_average(sizes, "VecAXPY", 3)
# plot_gpu_data_and_average(sizes, "VecSet", 3)
# plot_gpu_data_and_average(sizes, "VecCopy", 1)

# fig11_cpu(sm, "VecDot", 1, "VecAXPY", 3, True)
# fig11_cpu(md, "VecDot", 1, "VecAXPY", 3, True)

# fig11_gpu([1000, 10000, 100000], "VecDot", 1, "VecAXPY", 3, True, False)
# fig11_gpu([1000, 50000, 100000], "VecDot", 1, "VecAXPY", 3, True)


# gpu_error_bars([1000, 25000, 50000, 75000, 100000], "VecDot", 1)
# gpu_error_bars([1000, 25000, 50000, 75000, 100000], "VecAXPY", 3)
# gpu_error_bars_with_data(sm, "VecDot", 1)
# gpu_error_bars_with_data(sm, "VecAXPY", 3)

# gpu_error_bars_with_data_moving_average(sm + md, "VecDot", 1)
# gpu_error_bars_with_data_moving_average(sm, "VecAXPY", 3)
# gpu_error_bars_with_data_moving_average(md, "VecAXPY", 3)
gpu_error_bars_with_data_moving_average(lg, "VecAXPY", 3)






