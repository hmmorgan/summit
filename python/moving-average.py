import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Moving average for GPU data (generalize later)

# Pick a machine
machine = "GPU" # "CPU"
file_prefix = "vec_ops.n6_g1_c7_a1."
# file_path = "vec-ops-latency"
file_path = "waitforgpu-latency"

# machine = "CPU"
# file_prefix = "vec_ops.n2_g0_c21_p42."
# file_path = "cpu-flush-cache"

# Pick an operation 
# operation = "VecDot"
# count = 1
operation = "VecAXPY"
count = 3

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

plt.plot(vec_sizes, all_data, linestyle="none", marker=".", color="grey", label="Data")
plt.plot(internal_sizes, moving_average, color="black", label="Moving average")
plt.title(machine + " " + operation + " execution time", fontsize=12)
plt.xlabel("Vector size", fontsize=12)
plt.ylabel("Seconds", fontsize=12)
plt.legend(loc="upper left", fontsize=12, frameon=False)
# plt.xscale('log')
# plt.xlim([8e2, 1.2e7])
# ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
# ax.xaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
# plt.savefig("../plots/" + operation + "_" + machine + "_moving_average.png")
plt.show()
