import utils as ut
import matplotlib.pyplot as plt
import re

# compare GPU and CPU operation
def cpu_vs_gpu(operation, count, show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu = []
	cpu = []

	if operation == "VecDot":
		mem_scale = 1
	elif operation == "VecAXPY":
		mem_scale = 1.5

	for size in gpu_sizes:
		if operation == "VecDot":
			scale = 1
			gpu.append(scale*float(ut.get_floprate("../data/waitforgpu/vec_ops.n6_g1_c2_a1." + str(size) + ".718559", operation, False, count))) # need to get this data

		elif operation == "VecAXPY":
			scale = 1
			gpu.append(scale*float(ut.get_floprate("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", operation, False, count))) # need to get this data

	for size in cpu_sizes:
		if operation == "VecDot":
			scale = 1
		elif operation == "VecAXPY":
			scale = 1
		cpu.append(scale*float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, True, count)))
	
	# plot
	fig, left = plt.subplots()
	right = left.twinx()
	cm = plt.get_cmap('inferno')

	left.plot(cpu_sizes, cpu,         color=cm((1.*2)/4),    label="42 CPU cores " + operation)
	left.plot(gpu_sizes, gpu,         color=cm((1.*1-1)/4),  label="6 GPUs " + operation)
	
	plt.xlim([500, 2000000000])
	left.set_title("GPU vs CPU " + operation + " performance", fontsize=12)
	left.set_xlabel("Vector size", fontsize=12)
	left.set_ylabel("MFlops/second", fontsize=12)
	left.legend(loc="upper left", fontsize=12, ncol=1, frameon=False)
	left.set_xscale('log')
	left.set_yscale('log')
	right.set_yscale('log')
	right.get_yaxis().set_visible(False)
	left.set_ylim(top=10000000)
	right.set_ylim(top=10000000*mem_scale)
	right.set_ylim(bottom=20)
	left.set_ylim(bottom=20)
	plt.tight_layout()

	plt.savefig("../plots/" + operation + "_CPU_vs_GPU_siampp.png")
	if show: plt.show()

# compare GPU and CPU operation
def cpu_vs_gpu_copy(show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu_VecCopy = []
	cpu_VecCopy = []

	gpu_ToGpu = []

	for size in gpu_sizes:		
		scale = 2 # two memory access
		time = ut.get_time("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", "VecCopy", 1)
		gpu_VecCopy.append(scale*ut.calc_rate(size, time))
		
		# pinned memory
		scale = 1
		if gpu_sizes <= 100000:
			run_num = ".732319"
		else:
			run_num = ".715071"
		time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a1." + str(size) + run_num, "VecCUDACopyTo", 1)
		gpu_ToGpu.append(scale*ut.calc_rate(size, time))

	for size in cpu_sizes:
		scale = 2 # two memory access
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), "VecCopy", 1)
		cpu_VecCopy.append(scale*ut.calc_rate(size, time))

	# calculate peak rates in 8 Mbyes/second
	cpu_rate = 135*1e9
	gpu_rate = 900*1e9
	cpu_peak = (2*cpu_rate)/(8*1e6)
	gpu_peak = (6*gpu_rate)/(8*1e6)
	cpu_to_gpu_rate = 50*1e9
	cpu_to_gpu_peak = (6*cpu_to_gpu_rate)/(8*1e6)
	
	# plot
	fig, left = plt.subplots()
	right = left.twinx()
	cm = plt.get_cmap('inferno')

	right.plot(cpu_sizes, cpu_VecCopy, color=cm((1.*2)/4),    label="42 CPU cores VecCopy")
	right.plot(gpu_sizes, gpu_VecCopy, color=cm((1.*1-1)/4),  label="6 GPUs VecCopy")
	right.plot(gpu_sizes, gpu_ToGpu,   color=cm((1.*3)/4),    label="Copy to GPU")
	
	plt.plot(2000000000, gpu_peak, color=cm((1.*1-1)/4),        linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	plt.plot(2000000000, cpu_peak, color=cm((1.*2)/4),          linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	plt.plot(2000000000, cpu_to_gpu_peak, color=cm((1.*3)/4),   linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	
	plt.xlim([500, 2000000000])
	left.set_title("GPU vs CPU copy performance", fontsize=12)
	left.set_xlabel("Vector size", fontsize=12)
	left.set_ylabel("8 MBytes/second", fontsize=12)
	right.legend(loc="upper left", fontsize=12, ncol=1, frameon=False)
	# plt.legend(loc="upper left", fontsize=12, ncol=1, frameon=False)
	left.set_xscale('log')
	left.set_yscale('log')
	right.set_yscale('log')
	right.get_yaxis().set_visible(False)
	left.set_ylim(top=10000000)
	right.set_ylim(top=10000000)

	right.set_ylim(bottom=20)
	left.set_ylim(bottom=20)
	plt.tight_layout()

	plt.savefig("../plots/CPU_vs_GPU_copy_siampp.png")
	if show: plt.show()

cpu_vs_gpu("VecDot", 1, True)
cpu_vs_gpu("VecAXPY", 3, True)
cpu_vs_gpu_copy(True)
