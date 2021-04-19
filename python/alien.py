import utils as ut
import matplotlib.pyplot as plt

#################################

# what to plot
only_alien = False
operation = "VecAXPY" # "VecDot" # 
savename = "../plots/" + operation + "_alien_cpu_flush_cache_backwards.png"
line_num = 3 # 1 # 
buffer_ = 10
title = False

#################################

# get data
vec_size = []
data = []
alien = [10360000, 10390000, 10410000, 10440000, 10460000, 10470000, 10490000, 10510000, 10520000, 10540000, 10570000, 10590000] 

if only_alien:
	range_ = alien
else:
	range_ = range(10360000-(buffer_*10000), 10590001+(buffer_*10000), 10000)

for size in range_:
	# time = float(ut.get_time("../data/alien/vec_ops.n2_g0_c21_p1." + str(size) + ".659461", operation, line_num))
	time = float(ut.get_time("../data/cpu-flush-cache-alien/vec_ops.n2_g0_c21_p1." + str(size) + ".822998", operation, line_num))
	data.append(time)
	vec_size.append(size)
	# try:
	# 	# time = float(ut.get_time("../data/cpu-flush-cache-backwards-alien/vec_ops.n2_g0_c21_p1." + str(size) + ".834144", operation, line_num))
	# 	data.append(time)
	# 	vec_size.append(size)
	# except:
	# 	print("Didn't find it")

# plot
plt.plot(vec_size, data, linestyle="none", marker=".", color="grey")
plt.xlim([1e7, 1.09e7])
plt.ylim([.005, .02])

if title:
	# title and frame
	plt.title(operation + " CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
# else:
# 	# get rid of the frame
# 	frame1 = plt.gca()
# 	for spine in plt.gca().spines.values():
# 		spine.set_visible(False)
# 	# get rid of ticks
# 	frame1.axes.get_xaxis().set_visible(False)
# 	frame1.axes.get_yaxis().set_visible(False)
plt.tight_layout()

# plt.savefig(savename)
plt.show()
plt.gcf().clear()

# VecDot/VecAXPY vectors that make up the alien:
# 10360000
# 10390000
# 10410000
# 10440000
# 10460000
# 10470000
# 10490000
# 10510000
# 10520000
# 10540000
# 10570000
# 10590000

