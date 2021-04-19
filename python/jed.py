import utils as ut
import matplotlib.pyplot as plt

# compare GPU and CPU operation
def jed_cpu_vs_gpu(operation, count, show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu = []
	cpu = []

	gpu_time = [] 
	cpu_time = []

	gpu_VecCopy = []
	cpu_VecCopy = []

	gpu_VecCopy_time = []
	cpu_VecCopy_time = []

	gpu_ToGpu = []
	gpu_ToGpu_time = []

	if operation == "VecDot":
		mem_scale = 1
	elif operation == "VecAXPY":
		mem_scale = 1.5

	for size in gpu_sizes:
		if operation == "VecDot":
			scale = 1

			# operation time and floprate
			gpu_time.append(ut.get_time("../data/waitforgpu/vec_ops.n6_g1_c2_a1." + str(size) + ".718559", operation, count))
			gpu.append(scale*float(ut.get_floprate("../data/waitforgpu/vec_ops.n6_g1_c2_a1." + str(size) + ".718559", operation, False, count))) 
		elif operation == "VecAXPY":
			scale = 1

			gpu_time.append(ut.get_time("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", operation, count))
			gpu.append(scale*float(ut.get_floprate("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", operation, False, count))) 
		
		# GPU copy time and bandwidth
		scale = 2/mem_scale
		time = ut.get_time("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", "VecCopy", 1)
		gpu_VecCopy_time.append(time)
		gpu_VecCopy.append(scale*ut.calc_rate(size, time))
		
		# GPU to CPU time and bandwidth, pinned memory
		scale = 1/mem_scale
		if gpu_sizes <= 100000:
			run_num = ".732319"
		else:
			run_num = ".715071"
		time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a1." + str(size) + run_num, "VecCUDACopyTo", 1)
		gpu_ToGpu_time.append(time)
		gpu_ToGpu.append(scale*ut.calc_rate(size, time))


	for size in cpu_sizes:
		scale = 1
		# CPU operation time and bandwidth
		# cpu_time.append(ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, count))
		# cpu.append(scale*float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, True, count)))

		cpu_time.append(ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, count))
		cpu.append(scale*float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, True, count)))

		# CPU copy time and bandwidth
		scale = 2/mem_scale
		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecCopy", 1)
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), "VecCopy", 1)
		cpu_VecCopy_time.append(time)
		cpu_VecCopy.append(scale*ut.calc_rate(size, time))

	# calculate peak rates in 8 Mbyes/second
	cpu_rate = 135*1e9
	gpu_rate = 900*1e9
	cpu_peak = (2*cpu_rate)/(8*1e6)
	gpu_peak = (6*gpu_rate)/(8*1e6)
	
	# plot
	fig, left = plt.subplots()
	right = left.twinx()
	cm = plt.get_cmap('inferno')

	left.plot(cpu_time,         cpu,         color=cm((1.*2)/4),    label="42 CPU cores " + operation)
	left.plot(cpu_VecCopy_time, cpu_VecCopy, color=cm((1.*2)/4),    linestyle="dashed", label="42 CPU cores VecCopy")
	left.plot(gpu_ToGpu_time,   gpu_ToGpu,   color=cm((1.*3)/4),    linestyle="dashed", label="6 GPUs copy to GPU")
	left.plot(gpu_time,         gpu,         color=cm((1.*1-1)/4),  label="6 GPUs " + operation)
	left.plot(gpu_VecCopy_time, gpu_VecCopy, color=cm((1.*1-1)/4),  linestyle="dashed", label="6 GPUs VecCopy")

	left.set_title("GPU vs CPU " + operation + " performance", fontsize=12)
	left.set_xlabel("Execution time (seconds)", fontsize=12)
	left.set_ylabel("MFlops/second", fontsize=12)
	right.set_ylabel("8 MBytes/second", fontsize=12)
	left.legend(loc="lower right", fontsize=12, ncol=1, frameon=False) # markerfirst=False
	left.set_xscale('log')
	left.set_yscale('log')
	right.set_yscale('log')
	left.set_ylim([7, 1000000])
	right.set_ylim([7, 1000000*mem_scale])
	plt.xlim([1e-6, .2])
	plt.savefig("../plots/jed_" + operation + "_CPU_vs_GPU.png")
	plt.tight_layout()
	if show: plt.show()

jed_cpu_vs_gpu("VecDot", 1, True)
jed_cpu_vs_gpu("VecAXPY", 3, True)

