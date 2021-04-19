import utils as ut
import matplotlib.pyplot as plt

# CPU to GPU transfer performance
def VecCUDACopyTo(pinned, show):

	cpus = [1, 2, 4]
	sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	data = []
	scale = 1 # no memory movement

	if pinned == "":

		bandwidth = []
		for size in sizes:

			time = ut.get_time("../data/vec-ops/vec_ops.n1_g1_c42_a1." + str(size) + ".654911", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
			bandwidth.append(scale*ut.calc_rate(size, time))

		data.append(bandwidth)
		
		for cpu in cpus:
			bandwidth = []

			for size in sizes:
				time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".654914", "VecCUDACopyTo", 1)
				bandwidth.append(scale*ut.calc_rate(size, time))

			data.append(bandwidth)

	if pinned == "_pinned":

		bandwidth = []
		for size in sizes:

			time = ut.get_time("../data/pinned/vec_ops.n1_g1_c42_a1." + str(size) + ".713339", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
			bandwidth.append(scale*ut.calc_rate(size, time))

		data.append(bandwidth)
		
		for cpu in cpus:
			bandwidth = []

			for size in sizes:
				time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".715071", "VecCUDACopyTo", 1)
				bandwidth.append(scale*ut.calc_rate(size, time))

			data.append(bandwidth)

	# calculate peak rates in 8 Mbyes/second
	rate = 50*1e9
	gpu1_peak = rate/(8*1e6)
	gpu6_peak = (6*rate)/(8*1e6)

	# plot
	labels = ["1 MPI rank and 1 GPU", "1 MPI rank per GPU", "2 MPI ranks per GPU", "4 MPI ranks per GPU"]
	num = len(labels)
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	for i in range(num):
		ax.plot(sizes, data[i], marker="o", label=labels[i])

	plt.plot(2000000000, gpu1_peak, color="black", linestyle="none",  markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	plt.plot(2000000000, gpu6_peak, color="black", linestyle="none",  markersize="15", markeredgewidth=2, marker="_", clip_on=False)
	# plt.text(1500000000, gpu1_peak, "1 GPU peak", horizontalalignment='right', verticalalignment='center')
	plt.text(1700000000, gpu1_peak+1200, "1 GPU peak", horizontalalignment='right', verticalalignment='center')
	plt.text(1200000000, gpu6_peak, "6 GPU peak", horizontalalignment='right', verticalalignment='center')

	plt.xlim([500, 2000000000])
	if pinned == "":
		plt.title("CPU to GPU transfer performance", fontsize=12)
	if pinned == "_pinned":
		plt.title("CPU to GPU transfer performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("8 Mbytes/second", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	# plt.savefig("../plots/VecCUDACopyTo_" + pinned + ".png")
	if show: plt.show()

# CPU to GPU transfer performance
def VecCUDACopyTo_all(show):

	cpus = [1]#, 2, 4]
	sizes = [1000, 10000, 100000]#, 1000000, 10000000, 100000000, 1000000000]

	data = []
	data_pinned = []
	data_pinned_waitforgpu = []
	scale = 1 # no memory movement

	print "Non-pinned"
	bandwidth = []
	for size in sizes:

		time = ut.get_time("../data/vec-ops/vec_ops.n1_g1_c42_a1." + str(size) + ".654911", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
		bandwidth.append(scale*ut.calc_rate(size, time))

	data.append(bandwidth)
	
	for cpu in cpus:
		bandwidth = []

		for size in sizes:
			time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".654914", "VecCUDACopyTo", 1)
			print cpu
			print size
			print time
			bandwidth.append(scale*ut.calc_rate(size, time))

		data.append(bandwidth)

	print "Pinned"
	bandwidth = []
	for size in sizes:

		time = ut.get_time("../data/pinned/vec_ops.n1_g1_c42_a1." + str(size) + ".713339", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
		bandwidth.append(scale*ut.calc_rate(size, time))

	data_pinned.append(bandwidth)
	
	for cpu in cpus:
		bandwidth = []

		for size in sizes:
			time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".715071", "VecCUDACopyTo", 1)
			print cpu
			print size
			print time
			bandwidth.append(scale*ut.calc_rate(size, time))

		data_pinned.append(bandwidth)

	print "Pinned WaitForGPU()"
	bandwidth = []
	for size in sizes:

		time = ut.get_time("../data/pinned/vec_ops.n1_g1_c2_a1." + str(size) + ".732318", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
		bandwidth.append(scale*ut.calc_rate(size, time))

	data_pinned_waitforgpu.append(bandwidth)
	
	for cpu in cpus:
		bandwidth = []

		for size in sizes:
			time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".732319", "VecCUDACopyTo", 1)
			print cpu
			print size
			print time
			bandwidth.append(scale*ut.calc_rate(size, time))

		data_pinned_waitforgpu.append(bandwidth)

	# calculate peak rates in 8 Mbyes/second
	rate = 50*1e9
	gpu1_peak = rate/(8*1e6)
	gpu6_peak = (6*rate)/(8*1e6)

	# plot
	labels = ["1 MPI rank and 1 GPU", "1 MPI rank per GPU"]#, "2 MPI ranks per GPU", "4 MPI ranks per GPU"]
	# labels_pinned = ["Pinned 1 MPI rank and 1 GPU", "Pinned 1 MPI rank per GPU", "Pinned 2 MPI ranks per GPU", "Pinned 4 MPI ranks per GPU"]
	num = len(labels)
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	for i in range(num):
		ax.plot(sizes, data[i], marker="o", linestyle="dashed")

	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	for i in range(num):
		ax.plot(sizes, data_pinned[i], marker="o", label=labels[i])

	for i in range(num):
		ax.plot(sizes, data_pinned_waitforgpu[i], marker="o", linestyle="dotted")

	# plt.plot(2000000000, gpu1_peak, color="black", linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)#, label="1 GPU peak")
	# plt.plot(2000000000, gpu6_peak, color="black", linestyle="none", markersize="15", markeredgewidth=2, marker="_", clip_on=False)#, label="6 GPU peak")
	# plt.text(1700000000, gpu1_peak+1200, "1 GPU peak", horizontalalignment='right', verticalalignment='center')
	# plt.text(1500000000, gpu6_peak, "6 GPU peak", horizontalalignment='right', verticalalignment='center')

	# plt.xlim([500, 2000000000])

	plt.title("CPU to GPU transfer performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("8 Mbytes/second", fontsize=12)
	ax.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	# plt.savefig("../plots/VecCUDACopyTo_all.png")
	if show: plt.show()

# CPU to GPU transfer performance
def VecCUDACopyTo_comparison(comp, show):

	cpus = [1, 2, 4]
	sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
	sizes_ = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
	ones = [1, 1, 1, 1, 1, 1, 1, 1, 1]
	zeros = [0, 0, 0, 0, 0, 0, 0, 0, 0]

	data = []
	data_pinned = []
	scale = 1 # no memory movement

	bandwidth = []
	for size in sizes:

		time = ut.get_time("../data/vec-ops/vec_ops.n1_g1_c42_a1." + str(size) + ".654911", "VecCUDACopyTo",   1) # 1 GPU with 1 CPU
		bandwidth1 = scale*ut.calc_rate(size, time)
		time = ut.get_time("../data/pinned/vec_ops.n1_g1_c42_a1." + str(size) + ".713339", "VecCUDACopyTo",   1)
		bandwidth2 = scale*ut.calc_rate(size, time)

		if comp == "_ratio":
			bandwidth.append(bandwidth2/bandwidth1)

	data.append(bandwidth)
	
	for cpu in cpus:
		bandwidth = []

		for size in sizes:
			time = ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".654914", "VecCUDACopyTo", 1)
			bandwidth1 = scale*ut.calc_rate(size, time)
			time = ut.get_time("../data/pinned/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".715071", "VecCUDACopyTo", 1)
			bandwidth2 = scale*ut.calc_rate(size, time)
			
			if comp == "_ratio":
				bandwidth.append(bandwidth2/bandwidth1)

			if (size == 10000 or size == 100000000)  and cpu == 1:
				print size
				time  =  ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a" + str(cpu) + "." + str(size) + ".654914", "VecCUDACopyTo", 1)
				print scale*ut.calc_rate(size, time)
				time  =   ut.get_time("../data/pinned/vec_ops.n1_g1_c42_a1." + str(size) + ".713339", "VecCUDACopyTo",   1)
				print scale*ut.calc_rate(size, time)

		data.append(bandwidth)

	# plot
	labels = ["1 MPI rank and 1 GPU", "1 MPI rank per GPU", "2 MPI ranks per GPU", "4 MPI ranks per GPU"]
	num = len(labels)
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*(i+2))/(num+2)) for i in range(num)])

	for i in range(num):
		ax.plot(sizes, data[i], marker="o", label=labels[i])

	ax.plot(sizes_, ones, color="black", linestyle="dashed")
	plt.xlim([500, 2000000000])

	plt.title("CPU to GPU transfer performance", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Pinned memory/non-pinned memory", fontsize=12)
	ax.legend(loc="upper left", fontsize=12, frameon=False)
	plt.tight_layout()
	plt.xscale('log')
	# plt.savefig("../plots/VecCUDACopyTo_ratio.png")
	if show: plt.show()

VecCUDACopyTo("", True)
VecCUDACopyTo("_pinned", True)
VecCUDACopyTo_all(True)
VecCUDACopyTo_comparison("_ratio", True) 


