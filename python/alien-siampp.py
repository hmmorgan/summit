import utils as ut
import matplotlib.pyplot as plt

#################################

# what to plot
only_alien = False
savename = "../plots/alien_siampp.png"
buffer_ = 10
title = True

#################################

# get data
vecdot = []
vecaxpy = []
alien = [10360000, 10390000, 10410000, 10440000, 10460000, 10470000, 10490000, 10510000, 10520000, 10540000, 10570000, 10590000] 

if only_alien:
	vec_size = alien
else:
	vec_size = range(10360000-(buffer_*10000), 10590001+(buffer_*10000), 10000)

for size in vec_size:
	time = float(ut.get_time("../data/cpu-flush-cache-alien/vec_ops.n2_g0_c21_p1." + str(size) + ".822998", "VecDot", 1))
	vecdot.append(time)

	time = float(ut.get_time("../data/cpu-flush-cache-alien/vec_ops.n2_g0_c21_p1." + str(size) + ".822998", "VecAXPY", 3))
	vecaxpy.append(time)

# plot
plt.plot(vec_size, vecdot, linestyle="none", marker=".", color="black", label="VecDot")
plt.plot(vec_size, vecaxpy, linestyle="none", marker=".", color="grey", label="VecAXPY")
plt.xlim([1.01e7, 1.09e7])
plt.ylim([.007, .02])

if title:
	plt.title("CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
else:
	# get rid of the frame
	frame1 = plt.gca()
	for spine in plt.gca().spines.values():
		spine.set_visible(False)
	# get rid of ticks
	frame1.axes.get_xaxis().set_visible(False)
	frame1.axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig(savename)
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

