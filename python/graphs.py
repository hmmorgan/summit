import utils as ut
import matplotlib.pyplot as plt
import re

# compare GPU and CPU operation
def cpu_vs_gpu(operation, count, show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu = []
	cpu = []

	gpu_VecCopy = []
	cpu_VecCopy = []

	gpu_ToGpu = []

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
		
		scale = 2/mem_scale
		time = ut.get_time("../data/figures-2-7-8-9/vec_ops.n6_g1_c2_a1." + str(size) + ".668627", "VecCopy", 1)
		gpu_VecCopy.append(scale*ut.calc_rate(size, time))
		
		# pinned memory
		scale = 1/mem_scale
		if gpu_sizes <= 100000:
			run_num = ".732319"
		else:
			run_num = ".715071"
		time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a1." + str(size) + run_num, "VecCUDACopyTo", 1)
		gpu_ToGpu.append(scale*ut.calc_rate(size, time))

	for size in cpu_sizes:
		if operation == "VecDot":
			scale = 1
		elif operation == "VecAXPY":
			scale = 1
		# cpu.append(scale*float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, True, count)))
		# cpu.append(scale*float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767597", operation, True, count)))
		# cpu.append(scale*float(ut.get_floprate("../data/cpu-flush-cache-vecset/vec_ops.n2_g0_c21_p42." + str(size) + ".783275", operation, True, count)))
		cpu.append(scale*float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, True, count)))


		scale = 2/mem_scale # VecCopy
		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecCopy", 1)
		# time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767597", "VecCopy", 1)
		# time = ut.get_time("../data/cpu-flush-cache-vecset/vec_ops.n2_g0_c21_p42." + str(size) + ".783275", "VecCopy", 1)
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

	left.plot(cpu_sizes, cpu,         color=cm((1.*2)/4),    label="42 CPU cores " + operation)
	left.plot(cpu_sizes, cpu_VecCopy, color=cm((1.*2)/4),    linestyle="dashed", label="42 CPU cores VecCopy")
	left.plot(gpu_sizes, gpu_ToGpu,   color=cm((1.*3)/4),    linestyle="dashed", label="6 GPUs copy to GPU")
	left.plot(gpu_sizes, gpu,         color=cm((1.*1-1)/4),  label="6 GPUs " + operation)
	left.plot(gpu_sizes, gpu_VecCopy, color=cm((1.*1-1)/4),  linestyle="dashed", label="6 GPUs VecCopy")
	
	plt.plot(2000000000, gpu_peak, color=cm((1.*1-1)/4),        linestyle="none", markersize="15", markeredgewidth=2, marker="_", label="GPU copy peak", clip_on=False)
	plt.plot(2000000000, cpu_to_gpu_peak, color=cm((1.*3)/4),   linestyle="none", markersize="15", markeredgewidth=2, marker="_", label="CPU to GPU copy peak", clip_on=False)
	plt.plot(2000000000, cpu_peak, color=cm((1.*2)/4),          linestyle="none", markersize="15", markeredgewidth=2, marker="_", label="CPU copy peak", clip_on=False)
	
	plt.xlim([500, 2000000000])
	left.set_title("GPU vs CPU " + operation + " performance", fontsize=12)
	left.set_xlabel("Vector size", fontsize=12)
	left.set_ylabel("MFlops/second", fontsize=12)
	right.set_ylabel("8 MBytes/second", fontsize=12)
	left.legend(loc="lower right", fontsize=12, ncol=1, frameon=False)
	plt.legend(loc="upper left", fontsize=12, ncol=1, frameon=False)
	left.set_xscale('log')
	left.set_yscale('log')
	right.set_yscale('log')
	top_ = 1000000
	left.set_ylim(top=top_)
	right.set_ylim(top=top_*mem_scale)
	right.set_ylim(bottom=7)
	left.set_ylim(bottom=7)
	plt.tight_layout()

	plt.savefig("../plots/" + operation + "_CPU_vs_GPU.png")
	if show: plt.show()

def gpu_time(operation, count, show):

	sizes = [1000, 10000, 100000, 1000000, 10000000]

	gpu_time = []

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, count)
		gpu_time.append(1000000*time) # cheating so that y-axis units are 10^-6

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i+1)/num) for i in range(num)])

	plt.plot(sizes, gpu_time, marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation)
	plt.title("GPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper " + ("left" if operation=="VecAXPY" else "left"), fontsize=12, frameon=False)
	plt.xscale('log')
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/GPU_" + operation + "_time.png")
	if show: plt.show()

def gpu_time_2_ops(operation1, count1, operation2, count2, show):

	sizes = [1000, 10000, 100000, 1000000, 10000000]

	gpu_time1 = []
	gpu_time2 = []

	num_gpus = 1

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/waitforgpu/vec_ops.n6_g1_c7_a1." + str(size) + ".715223", operation1, count1)
		# time = ut.get_time("../data/waitforgpu/vec_ops.n1_g1_c2_a1." + str(size) + ".718552", operation1, count1)
		gpu_time1.append(1000000*time) # cheating so that y-axis units are 10^-6

		time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation2, count2)
		# time = ut.get_time("../data/vec-ops/vec_ops.n1_g1_c2_a1." + str(size) + ".654909", operation2, count2)
		gpu_time2.append(1000000*time) # cheating so that y-axis units are 10^-6

	print gpu_time1[0]
	print gpu_time2[0]

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	# plt.plot(sizes, gpu_time1, marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation1)
	# plt.plot(sizes, gpu_time2, marker=".", markersize="6", markeredgewidth=2, label="6 GPUs " + operation2)
	plt.plot(sizes, gpu_time1, marker=".", markersize="6", markeredgewidth=2, label="1 GPU " + operation1)
	plt.plot(sizes, gpu_time2, marker=".", markersize="6", markeredgewidth=2, label="1 GPU " + operation2)
	plt.title("GPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	# plt.xscale('log')
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	# plt.savefig("../plots/GPU_" + operation1 + "_" + operation2 + "_time.png")
	plt.savefig("../plots/GPU_" + operation1 + "_" + operation2 + "_time_1GPU_.png")
	if show: plt.show()

def cpu_time(operation, count, show):

	sizes = [1000, 10000, 100000]

	cpu_time = []

	# get data
	for size in sizes:

		# time from file
		time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p1." + str(size) + ".654910", operation, count)
		cpu_time.append(1000000*time) # cheating so that y-axis units are 10^-6

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i+1)/num) for i in range(num)])

	plt.plot(sizes, cpu_time, marker=".", markersize="6", markeredgewidth=2, label="1 CPU " + operation)
	plt.title("CPU " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper " + ("left" if operation=="VecAXPY" else "left"), fontsize=12, frameon=False)
	plt.xscale('log')
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/CPU_" + operation + "_time.png")
	if show: plt.show()

def cpu_time_2_ops(operation1, count1, operation2, count2, show):

	sizes = [1000, 10000, 100000, 1000000]

	cpu_time1 = []
	cpu_time2 = []
	cpu_time3 = []

	# get data
	for size in sizes:

		# time from file
		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation1, count1)
		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p1." + str(size) + ".654910", operation1, count1)
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation1, count1)
		cpu_time1.append(1000000*time) # cheating so that y-axis units are 10^-6

		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation2, count2)
		# time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p1." + str(size) + ".654910", operation2, count2)
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation2, count2)
		cpu_time2.append(1000000*time) # cheating so that y-axis units are 10^-6

		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), "VecCopy", 1)
		cpu_time3.append(1000000*time) # cheating so that y-axis units are 10^-6

	print cpu_time1[0]
	print cpu_time2[0]
	print cpu_time3[0]

	# plot
	num = 4
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	plt.plot(sizes, cpu_time1, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs " + operation1)
	plt.plot(sizes, cpu_time2, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs " + operation2)
	plt.plot(sizes, cpu_time3, marker=".", markersize="6", markeredgewidth=2, label="42 CPUs VecCopy")
	# plt.plot(sizes, cpu_time1, marker=".", markersize="6", markeredgewidth=2, label="1 CPU " + operation1)
	# plt.plot(sizes, cpu_time2, marker=".", markersize="6", markeredgewidth=2, label="1 CPU " + operation2)
	plt.title("CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	# plt.xscale('log')
	ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	# plt.ylim([0, 350])
	plt.tight_layout()

	plt.savefig("../plots/CPU_" + operation1 + "_" + operation2 + "_time_42CPU.png")
	# plt.savefig("../plots/CPU_" + operation1 + "_" + operation2 + "_time_1CPU.png")
	if show: plt.show()

# GPU flop rate
def gpu_flops(operation, count, show):

	gpus = range(1, 7)
	sizes = [100000, 1000000, 10000000, 100000000, 1000000000]
	sizes_str = ["$10^5$", "$10^6$", "$10^7$", "$10^8$", "$10^9$"]

	# get flop rates
	data = []

	for size in sizes:
		flop_rates = []

		for gpu in gpus:
			if operation == "VecDot":
				rate = ut.get_floprate("../data/waitforgpu/vec_ops.n" + str(gpu) + "_g1_c2_a1." + str(size) + ".718552", operation, False, count)
				flop_rates.append(float(rate))

			elif operation == "VecAXPY":
				rate = ut.get_floprate("../data/vec-ops/vec_ops.n" + str(gpu) + "_g1_c2_a1." + str(size) + ".654909", operation, False, count)
				flop_rates.append(float(rate))

		data.append(flop_rates)

	# plot
	num = len(sizes)
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	for i in range(num):
		ax.plot(gpus, data[i], marker="o", label="Vec size " + sizes_str[i])
	
	plt.title(operation + " GPU performance", fontsize=12)
	plt.xlabel("Number of GPUs", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.xlim([0, 7])
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
	plt.tight_layout()

	plt.savefig("../plots/GPU_" + operation + "_flops.png")
	if show: plt.show()

# CPU flop rate
def cpu_flops(operation, count, show):

	ranks = range(1, 43)
	vecsize = [10000, 1000000, 100000000]
	vecsize_str = ["$10^4$", "$10^6$", "$10^8$"]

	# get flop rates
	data = []

	for size in vecsize:
		flop_rates = []

		for rank in ranks:
			# rate = ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p" + str(rank) + "." + str(size) + ".654910", operation, True, count)
			rate = ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p" + str(rank) + "." + str(size), operation, True, count)

			flop_rates.append(float(rate))

		data.append(flop_rates)

	# plot
	num = len(vecsize) 
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	for i in range(num):
		ax.plot(ranks, data[i], marker="o", linestyle="none", label="Vec size " + vecsize_str[i])

	plt.title(operation + " CPU performance", fontsize=12)
	plt.xlabel("Number of CPU cores", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xlim([-3, 43])
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	plt.savefig("../plots/CPU_" + operation + "_flops.png")
	if show: plt.show()

# GPU virtualization on Summit
def gpu_virtualization(operation, count, height, show):

	gpu1 = []
	gpu2 = []
	gpu3 = []
	gpu6 = []

	size = 100000000
	size_str = "10^8"

	if operation == "VecDot":
		for cpus in range(1, 42):
			gpu1.append(float(ut.get_floprate("../data/waitforgpu/vec_ops.n1_g1_c42_a" + str(cpus) + "." + str(size) + ".718553", operation, False, count)))

		for cpus in range(1, 22):
			gpu2.append(float(ut.get_floprate("../data/waitforgpu/vec_ops.n2_g1_c21_a" + str(cpus) + "." + str(size) + ".718554", operation, False, count)))

		for cpus in range(1, 15):
			gpu3.append(float(ut.get_floprate("../data/waitforgpu/vec_ops.n3_g1_c14_a" + str(cpus) + "." + str(size) + ".718555", operation, False, count)))

		for cpus in range(1, 8):
			gpu6.append(float(ut.get_floprate("../data/waitforgpu/vec_ops.n6_g1_c7_a"  + str(cpus) + "." + str(size) + ".718556", operation, False, count)))

	elif operation == "VecAXPY":
		for cpus in range(1, 42):
			gpu1.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(cpus) + "." + str(size) + ".654911", operation, False, count)))

		for cpus in range(1, 22):
			gpu2.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g1_c21_a" + str(cpus) + "." + str(size) + ".654912", operation, False, count)))

		for cpus in range(1, 15):
			gpu3.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n3_g1_c14_a" + str(cpus) + "." + str(size) + ".654913", operation, False, count)))

		for cpus in range(1, 8):
			gpu6.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n6_g1_c7_a"  + str(cpus) + "." + str(size) + ".654914", operation, False, count)))

	# plot
	num = 4
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	ax.plot(range(1, 42), gpu1,    marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="1 GPU")
	ax.plot(range(2, 43, 2), gpu2, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="2 GPUs")
	ax.plot(range(3, 43, 3), gpu3, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="3 GPUs")
	ax.plot(range(6, 43, 6), gpu6, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="6 GPUs")

	plt.title(operation + " virtualization performance", fontsize=12)
	plt.xlabel("MPI ranks", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="upper right", ncol=2, fontsize=12, frameon=False)
	plt.xlim([0, 43])
	plt.ylim(top=height)
	plt.tight_layout()
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	plt.savefig("../plots/" + operation + "_virtualization_" + size_str + ".png")
	if show: plt.show()

# compare GPU and CPU flops
def cpu_gpu_flops(operation, count, normed, ncols, show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	gpu1 = []
	gpu2 = []
	gpu3 = []
	gpu6 = []

	cpu7  = []
	cpu14 = []
	cpu21 = []
	cpu28 = []
	cpu35 = []
	cpu42 = []

	if operation == "VecDot":
		path = "waitforgpu"
		run_num = ".718559"
	elif operation == "VecAXPY":
		path = "figures-2-7-8-9"
		run_num = ".668627"

	for size in gpu_sizes:
		gpu1.append(float(ut.get_floprate("../data/" + path + "/vec_ops.n1_g1_c2_a1." + str(size) + run_num, operation, False, count)))
		gpu2.append(float(ut.get_floprate("../data/" + path + "/vec_ops.n2_g1_c2_a1." + str(size) + run_num, operation, False, count)))
		gpu3.append(float(ut.get_floprate("../data/" + path + "/vec_ops.n3_g1_c2_a1." + str(size) + run_num, operation, False, count)))
		gpu6.append(float(ut.get_floprate("../data/" + path + "/vec_ops.n6_g1_c2_a1." + str(size) + run_num, operation, False, count)))

	for size in cpu_sizes:
		# cpu7.append( float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p7."  + str(size) + ".654910", operation, True, count)))
		# cpu14.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p14." + str(size) + ".654910", operation, True, count)))
		# cpu21.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p21." + str(size) + ".654910", operation, True, count)))
		# cpu28.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p28." + str(size) + ".654910", operation, True, count)))
		# cpu35.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p35." + str(size) + ".654910", operation, True, count)))
		# cpu42.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, True, count)))
		cpu7.append( float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p7."  + str(size), operation, True, count)))
		cpu14.append(float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p14." + str(size), operation, True, count)))
		cpu21.append(float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p21." + str(size), operation, True, count)))
		cpu28.append(float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p28." + str(size), operation, True, count)))
		cpu35.append(float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p35." + str(size), operation, True, count)))
		cpu42.append(float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, True, count)))

	if normed:
		gpu2[:] = [x/2.0 for x in gpu2]
		gpu3[:] = [x/3.0 for x in gpu3]
		gpu6[:] = [x/6.0 for x in gpu6]

		cpu7[:] =  [x/7.0  for x in cpu7]
		cpu14[:] = [x/14.0 for x in cpu14]
		cpu21[:] = [x/21.0 for x in cpu21]
		cpu28[:] = [x/28.0 for x in cpu28]
		cpu35[:] = [x/35.0 for x in cpu35]
		cpu42[:] = [x/42.0 for x in cpu42]

	# plot
	num = 10
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	ax.plot(cpu_sizes, cpu7,  marker="o", markersize="4", markeredgewidth=2, label= "7 CPU cores")
	ax.plot(cpu_sizes, cpu14, marker="o", markersize="4", markeredgewidth=2, label="14 CPU cores")
	ax.plot(cpu_sizes, cpu21, marker="o", markersize="4", markeredgewidth=2, label="21 CPU cores")
	ax.plot(cpu_sizes, cpu28, marker="o", markersize="4", markeredgewidth=2, label="28 CPU cores")
	ax.plot(cpu_sizes, cpu35, marker="o", markersize="4", markeredgewidth=2, label="35 CPU cores")
	ax.plot(cpu_sizes, cpu42, marker="o", markersize="4", markeredgewidth=2, label="42 CPU cores")

	ax.plot(gpu_sizes, gpu1, marker="o", markersize="4", markeredgewidth=2, label="1 GPU")
	ax.plot(gpu_sizes, gpu2, marker="o", markersize="4", markeredgewidth=2, label="2 GPUs")
	ax.plot(gpu_sizes, gpu3, marker="o", markersize="4", markeredgewidth=2, label="3 GPUs")
	ax.plot(gpu_sizes, gpu6, marker="o", markersize="4", markeredgewidth=2, label="6 GPUs")

	plt.title(operation + " performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, ncol=ncols, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
		
	plt.savefig("../plots/CPU_GPU_" + operation + "_flops" + ("_norm" if normed else "") + ".png")
	if show: plt.show()

# VecSet performance
def VecSet(show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	cpu = []
	gpu = []

	for size in cpu_sizes:
		time1 = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p7." + str(size) + ".654910", "VecSet", 2) # 7 CPUs
		cpu.append(ut.calc_rate(size, time1))

	for size in gpu_sizes:
		time2 = ut.get_time("../data/figures-2-7-8-9/vec_ops.n1_g1_c2_a1." + str(size) + ".668627", "VecSet", 3) # 1 GPU with 1 CPU
		gpu.append(ut.calc_rate(size, time2))

	# plot
	num = 2
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	ax.plot(cpu_sizes, cpu, marker="o", markersize="4", markeredgewidth=2, label="7 CPU cores")
	ax.plot(gpu_sizes, gpu, marker="o", markersize="4", markeredgewidth=2, label="1 GPU")

	plt.title("VecSet performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("8 Mbytes/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	plt.savefig("../plots/VecSet.png")
	if show: plt.show()

# VecCopy performance
def VecCopy(show):

	gpu_sizes = [1000, 10000, 100000, 1000000, 2000000, 4000000, 6000000, 8000000, 10000000, 20000000, 40000000, 60000000, 80000000, 100000000, 1000000000]
	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	cpu = []
	gpu = []
	cputogpu = []
	scale = 2 # for VecCopy

	for size in cpu_sizes:
		time1 = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p7." + str(size) + ".654910", "VecCopy", 1)
		cpu.append(ut.calc_rate(size, time1))	

	
	for size in gpu_sizes:
		scale = 2 # for VecCopy
		time2 = ut.get_time("../data/figures-2-7-8-9/vec_ops.n1_g1_c2_a1." + str(size) + ".668627", "VecCopy", 1)
		gpu.append(scale*ut.calc_rate(size, time2))

		scale = 1 # for copy to GPU
		nonpinned = ut.get_time("../data/figures-2-7-8-9/vec_ops.n1_g1_c2_a1." + str(size) + ".668627", "VecCUDACopyTo", 1)
		pinned = ut.get_time("../data/pinned/vec_ops.n1_g1_c42_a1." + str(size) + ".720947", "VecCUDACopyTo", 1)
		if pinned < nonpinned:
			time3 = pinned
		else:
			time3 = nonpinned
		
		cputogpu.append(scale*ut.calc_rate(size, time3))

	# calculate peak rates in 8 Mbytes/s
	cpu_rate = 135*1e9
	gpu_rate = 900*1e9
	cpu_to_gpu_rate = 50*1e9

	cpu_peak = (2*cpu_rate)/(8*1e6)
	gpu_peak = gpu_rate/(8*1e6)
	cpu_to_gpu_peak = cpu_to_gpu_rate/(8*1e6)

	# plot
	num = 4
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	ax.plot(cpu_sizes, cpu, marker="o", markersize="4", markeredgewidth=2, label="7 CPU cores VecCopy")
	ax.plot(gpu_sizes, gpu, marker="o", markersize="4", markeredgewidth=2, label="1 GPU VecCopy")
	ax.plot(gpu_sizes, cputogpu, marker="o", markersize="4", markeredgewidth=2, label="1 GPU copy to GPU")

	plt.plot(2000000000, gpu_peak, color="black", linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	plt.plot(2000000000, cpu_peak, color="black", linestyle="none", markersize="15", markeredgewidth=2, marker="_",  clip_on=False)
	plt.plot(2000000000, cpu_to_gpu_peak, color="black", linestyle="none", markersize="15", markeredgewidth=2, marker="_",  clip_on=False)

	# print cputogpu[-1]
	# print cpu_to_gpu_peak


	plt.text(1200000000, gpu_peak, "GPU copy peak", horizontalalignment='right', verticalalignment='center')
	plt.text(1200000000, cpu_peak, "CPU copy peak", horizontalalignment='right', verticalalignment='center')
	plt.text(1600000000, cpu_to_gpu_peak-6000, "CPU to GPU peak", horizontalalignment='right', verticalalignment='center')

	plt.xlim([500, 2000000000])
	plt.title("VecCopy performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("8 Mbytes/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	plt.savefig("../plots/VecCopy.png")
	if show: plt.show()

# make plots
# cpu_vs_gpu("VecDot", 1, True)
# cpu_vs_gpu("VecAXPY", 3, True)
# gpu_time("VecDot", 1, True)
# gpu_time("VecAXPY", 3, True)
gpu_time_2_ops("VecDot", 1, "VecAXPY", 3, True)
# cpu_time("VecDot", 1, True)
# cpu_time("VecAXPY", 3, True)
cpu_time_2_ops("VecDot", 1, "VecAXPY", 3, True)
# gpu_flops("VecDot", 1, True)
# gpu_flops("VecAXPY", 3, False)
# cpu_flops("VecDot", 1, True)
# cpu_flops("VecAXPY", 3, True)
# gpu_virtualization("VecDot", 1, 650000, True)
# gpu_virtualization("VecAXPY", 3, 450000, False)
# cpu_gpu_flops("VecDot", 1, True, 1, True)
# cpu_gpu_flops("VecDot", 1, False, 1, True)
# cpu_gpu_flops("VecAXPY", 3, True, 1, True)
# cpu_gpu_flops("VecAXPY", 3, False, 1, True)
# VecSet(True)
# VecCopy(True)





