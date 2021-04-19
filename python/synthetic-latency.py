import utils as ut
import matplotlib.pyplot as plt
import numpy as np

def synthetic_latency(operation, count, show):

	# get data
	sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu = []
	cpu = []

	gpu_VecCopy = []
	cpu_VecCopy = []

	gpu_ToGpu = []

	gpu_16 = []
	gpu_28_time = []
	gpu_24 = []
	gpu_28 = []

	for size in sizes:

		# floprate from file
		floprate = float(ut.get_floprate("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, False, count))
		gpu.append(floprate)

		# time from file
		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, count)

		# create synthetic floprates
		if operation == "VecAXPY":
			labels = ["16", "24", "28"]
			gpu_16.append((2*size*1e-6)/(time-16e-6))
			gpu_24.append((2*size*1e-6)/(time-24e-6))
			gpu_28.append((2*size*1e-6)/(time-28e-6))
		elif operation == "VecDot":
			labels = ["16", "24", "50"]
			gpu_16.append((2*size*1e-6)/(time-16e-6))
			gpu_24.append((2*size*1e-6)/(time-24e-6))
			gpu_28.append((2*size*1e-6)/(time-50e-6)) # VecDot bigger latencies in data

		# other operations	
		cpu.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, True, count)))

		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", "VecCopy", 1)
		gpu_VecCopy.append(ut.calc_rate(size, time))
		time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecCopy", 1)
		cpu_VecCopy.append(ut.calc_rate(size, time))

		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", "VecCUDACopyTo", 1)
		gpu_ToGpu.append(ut.calc_rate(size, time))

	# plot
	plt.plot(sizes, cpu,         color="grey", alpha=0.5, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs " + operation)
	plt.plot(sizes, gpu,         color="black",  marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation)
	plt.plot(sizes, gpu_24,      color="black",  marker=".", markersize="6", markeredgewidth=2, linestyle="dotted",label="$" + labels[1] + "\cdot10^{-6}$ latency")

	plt.title(operation + " performance without calculated latency", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="lower right", fontsize=12, frameon=False)
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim(top=1000000)
	plt.tight_layout()

	plt.savefig("../plots/" + operation + "_synthetic_latency.png")
	if show: plt.show()
	plt.gcf().clear()


def synthetic_time(operation, count, show):

	sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu_time = []
	gpu_16 = []
	gpu_24 = []
	gpu_28 = []

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, count)
		gpu_time.append(time)

		# create synthetic floprates
		if operation == "VecAXPY":
			labels = ["16", "24", "28"]
			gpu_16.append(time-16e-6)
			gpu_24.append(time-24e-6)
			gpu_28.append(time-28e-6)
		elif operation == "VecDot":
			labels = ["16", "24", "50"]
			gpu_16.append(time-16e-6)
			gpu_24.append(time-24e-6)
			gpu_28.append(time-50e-6)

	# plot
	plt.plot(sizes, gpu_time,    color="black",  marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation)
	plt.plot(sizes, gpu_16,      color="black",  marker=".", markersize="6", markeredgewidth=2, linestyle="dashed", label="$" + labels[0] + "\cdot10^{-6}$ latency")
	plt.plot(sizes, gpu_24,      color="black",  marker=".", markersize="6", markeredgewidth=2, linestyle="dashdot",label="$" + labels[1] + "\cdot10^{-6}$ latency")
	plt.plot(sizes, gpu_28,      color="black",  marker=".", markersize="6", markeredgewidth=2, linestyle="dotted", label="$" + labels[2] + "\cdot10^{-6}$ latency")

	plt.title("GPU vs CPU " + operation + " performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	plt.yscale('log')

	plt.savefig("../plots/" + operation + "_synthetic_time.png")
	if show: plt.show()
	plt.gcf().clear()

# make graphs
synthetic_latency("VecAXPY", 3, False)
synthetic_latency("VecDot", 1, False)
# synthetic_time("VecAXPY", 3, True)
# synthetic_time("VecDot", 1, True)

