import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Average for GPU data

machine = "GPU"
file_prefix = "vec_ops.n6_g1_c7_a1"
file_path = "gpu-statistics"

# Pick an operation 
# operation = "VecDot"
# count = 1
operation = "VecAXPY"
count = 3

# operation = "VecSet"
# count = 3

# operation =  "VecCopy"
# count = 1

# Get the data
sm = range(1000, 100001, 1000)
md = range(100000, 10000001, 100000) 
lg = range(10000000, 100000001, 1000000)
vec_sizes = sm + md + lg

# vec_sizes = [1000, 2000]

num_samples = 10

all_data = []
for size in vec_sizes:
	x = []
	for sample in range(1, num_samples + 1):
		file = "../data/{}/{}.{}.{}".format(file_path, file_prefix, size, sample)
		x.append(float(ut.get_time(file, operation, count)))
	all_data.append(x)

average = []
for i in range(len(vec_sizes)):
	curr = all_data[i]
	average.append(sum(curr)/num_samples)

# Make the plot
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(vec_sizes)):
	if i == 0:
		plt.plot([vec_sizes[i]]*10, all_data[i], linestyle="none", marker=".", color="grey", label="Data")
	else:
		plt.plot([vec_sizes[i]]*10, all_data[i], linestyle="none", marker=".", color="grey")
# plt.plot(vec_sizes, average, color="black", label="Mean")
plt.title(machine + " " + operation + " execution time", fontsize=12)
plt.xlabel("Vector size", fontsize=12)
plt.ylabel("Seconds", fontsize=12)
plt.legend(loc="upper right", fontsize=12, frameon=False)
# plt.xlim([8e2, 1.2e5])
ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))

plt.tight_layout()
# plt.savefig("../plots/" + operation + "_" + machine + "_mean.png")
plt.show()

# print operation
# print average[0], average[1], average[2]

