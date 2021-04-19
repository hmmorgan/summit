import utils as ut
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
vec_size = [1e3, 1e5, 1e7, 1e9]
vec_size_string = ["10_3", "10_5", "10_7", "10_9"]

# means = []
# std_devs = []
all_data = []
data = []
for size in vec_size_string:
	for i in range(1, 11):
		data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + size + "." + str(i), operation, count)))
	# means.append(scipy.mean(data))
	# std_devs.append(np.std(data))
	# data = []
	all_data.append(data)
	data = []


# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(vec_size)):
	plt.plot([vec_size[i]]*10, all_data[i], linestyle="none", marker=".", color="grey")

# plt.errorbar(vec_size, means, yerr=std_devs, linestyle="none")

plt.title(machine + " " + operation + " execution time", fontsize=12)
plt.xlabel("Vector size", fontsize=12)
plt.ylabel("Seconds", fontsize=12)
plt.legend(loc="upper left", fontsize=12, frameon=False)
plt.xscale('log')
plt.xlim([8e2, 1.2e7])
# ax.ticklabel_format(axis="both", style="sci", scilimits=(0.1,0))
# ax.xaxis.set_major_locator(plt.MaxNLocator(5))

plt.tight_layout()
# plt.savefig("../plots/" + operation + "_" + machine + "_moving_average.png")
plt.show()
