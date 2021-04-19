import utils as ut
import matplotlib.pyplot as plt

def gpu_time(operation, count, show):

	sizes = [1000, 10000, 100000, 1000000, 10000000]#, 100000000]#, 1000000000]

	gpu_time1 = []
	gpu_time2 = []

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, count)
		gpu_time1.append(1000000*time) # cheating so that y-axis units are 10^-6

		# removes WaitForGpu() from VecDot_SeqCUDA()
		time = ut.get_time("../data/vecdot-waitforgpu/vec_ops.n6_g1_c7_a1." + str(size) + ".715223", operation, count)
		gpu_time2.append(1000000*time)

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	plt.plot(sizes, gpu_time1, marker=".", markersize="6", markeredgewidth=2, label=operation)
	plt.plot(sizes, gpu_time2, marker=".", markersize="6", markeredgewidth=2, label=operation + " without WaitForGPU()")

	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper " + ("left" if operation=="VecAXPY" else "left"), fontsize=12, frameon=False)
	plt.xscale('log')
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/GPU_" + operation + "_time_waitforgpu.png")
	if show: plt.show()

gpu_time("VecDot", 1, True)