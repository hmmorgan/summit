import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# figure 2
def cpu_vs_gpu(operation, count, clear, show):

	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	cpu_flush = []
	cpu_half_flush = []
	cpu_no_flush = []
	cpu_permute = []
	cpu_flush_vecset = []
	cpu_half_flush_december = []

	cpu_VecCopy_flush = []
	cpu_VecCopy_half_flush = []
	cpu_VecCopy_no_flush = []

	if operation == "VecDot":
		mem_scale = 1
	elif operation == "VecAXPY":
		mem_scale = 1.5

	for size in cpu_sizes:
		if operation == "VecDot":
			scale = 1
		elif operation == "VecAXPY":
			scale = 1
		cpu_flush.append(scale*float(ut.get_floprate("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767597", operation, True, count)))
		cpu_half_flush.append(scale*float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", operation, True, count)))
		cpu_half_flush_december.append(scale*float(ut.get_floprate("../data/vec-ops-december/vec_ops.n2_g0_c21_p42." + str(size) + ".795805", operation, True, count)))
		cpu_no_flush.append(scale*float(ut.get_floprate("../data/cpu-no-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767590", operation, True, count)))
		cpu_flush_vecset.append(scale*float(ut.get_floprate("../data/cpu-flush-cache-vecset/vec_ops.n2_g0_c21_p42." + str(size) + ".792547", operation, True, count)))
		cpu_permute.append(scale*float(ut.get_floprate("../data/permute-operations/vec_ops.n2_g0_c21_p42." + str(size) + ".792549", operation, True, count)))


		scale = 2/mem_scale # VecCopy
		time = ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767597", "VecCopy", 1)
		cpu_VecCopy_flush.append(scale*ut.calc_rate(size, time))
		time = ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecCopy", 1)
		cpu_VecCopy_half_flush.append(scale*ut.calc_rate(size, time))
		time = ut.get_time("../data/cpu-no-flush-cache/vec_ops.n2_g0_c21_p42." + str(size) + ".767590", "VecCopy", 1)
		cpu_VecCopy_no_flush.append(scale*ut.calc_rate(size, time))

	print cpu_half_flush_december[0]
	
	# plot
	fig, left = plt.subplots()
	right = left.twinx()
	cm = plt.get_cmap('inferno')

	left.plot(cpu_sizes, cpu_flush, color=cm((1.*2)/4), label=operation + " cleared cache")
	left.plot(cpu_sizes, cpu_half_flush, color=cm((1.*2)/4), linestyle="dashed", label=operation+ " half cleared cache")
	left.plot(cpu_sizes, cpu_half_flush_december, color="black", label=operation+ " December")
	left.plot(cpu_sizes, cpu_no_flush, color=cm((1.*2)/4), linestyle="dotted", label=operation+ " uncleared cache")
	# left.plot(cpu_sizes, cpu_permute, color="black", linestyle="dashed", label=operation+ " another cleared")
	# left.plot(cpu_sizes, cpu_permute, color="black", label=operation+ " rearrange operations")

	# left.plot(cpu_sizes, cpu_VecCopy_flush, color=cm((1.*1-1)/4), label="VecCopy cleared cache")
	# left.plot(cpu_sizes, cpu_VecCopy_half_flush, color=cm((1.*1-1)/4), linestyle="dashed", label="VecCopy half cleared cache")
	# left.plot(cpu_sizes, cpu_VecCopy_no_flush, color=cm((1.*1-1)/4), linestyle="dotted", label="VecCopy uncleared cache")
	
	plt.xlim([500, 2000000000])
	left.set_title("CPU " + operation + " cache performance", fontsize=12)
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
	right.set_ylim(bottom=20)
	left.set_ylim(bottom=20)
	plt.tight_layout()

	# plt.savefig("../plots/" + operation + "_CPU_cleared_cache.png")
	if show: plt.show()

cpu_vs_gpu("VecDot", 1, True, True)
cpu_vs_gpu("VecAXPY", 3, True, True)

def plot(machine, operation, count, whole_domain, show):

	# get data
	sm = range(1000, 100001, 100)
	md = range(100000, 10000001, 10000) 
	lg = range(10000000, 100000001, 100000)

	sm_data = []
	md_data = []
	lg_data = []


	if machine == "CPU":
		file_path = "cpu-flush-cache-vecset-latency"
		file_prefix = "vec_ops.n2_g0_c21_p42."

	# operation = "VecAXPY"
	# count = 3

	for size in sm:
		sm_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))
	for size in md:
		md_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))
	for size in lg:
		lg_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))

	sm = np.array(sm)
	md = np.array(md)
	lg = np.array(lg)

	sm_data = np.array(sm_data)
	md_data = np.array(md_data)
	lg_data = np.array(lg_data)

	# prune NaNs
	mask1 = np.isfinite(sm) & np.isfinite(sm_data)
	mask2 = np.isfinite(md) & np.isfinite(md_data)
	mask3 = np.isfinite(lg) & np.isfinite(lg_data)

	s1, i1, r, p, stderr = stats.linregress(sm[mask1], sm_data[mask1])
	s2, i2, r, p, stderr = stats.linregress(md[mask2], md_data[mask2])
	s3, i3, r, p, stderr = stats.linregress(lg[mask3], lg_data[mask3]) 

	if whole_domain:
		whole = np.array(range(1000, 100000001, 10000))
		y1 = s1*whole + i1
		y2 = s2*whole + i2 
		y3 = s3*whole + i3 
	else: 
		y1 = s1*sm + i1
		y2 = s2*md + i2 
		y3 = s3*lg + i3

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(sm, sm_data, linestyle="none", marker=".", color="grey")
	plt.plot(md, md_data, linestyle="none", marker=".", color="grey")
	plt.plot(lg, lg_data, linestyle="none", marker=".", color="grey")

	if whole_domain:
		plt.plot(whole, y1, color="black", label="Fitted for vectors in $10^3$ - $10^5$", linestyle="dotted")
		plt.plot(whole, y2, color="black", label="Fitted for vectors in $10^5$ - $10^7$", linestyle="dashed")  
		plt.plot(whole, y3, color="black", label="Fitted for vectors in $10^7$ - $10^8$")  
	# else:
	# 	plt.plot(sm, y1, color="black")
	# 	plt.plot(md, y2, color="black")
	# 	plt.plot(lg, y3, color="black")

	plt.title(machine + " " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
	# ax.xaxis.set_major_locator(plt.MaxNLocator(5))

	plt.tight_layout()
	# plt.savefig("../plots/" + operation + "_" + machine + "_latency_42CPU.png") #VecDot_CPU_latency
	if show: plt.show()

plot("CPU", "VecDot", 1, True, True)
plot("CPU", "VecCopy", 1, True, True)
plot("CPU", "VecDot", 1, True, True)
plot("CPU", "VecAXPY", 3, True, True)

def table(machine):

	# get data
	sm = range(1000, 100001, 100)
	md = range(100000, 10000001, 10000) 
	lg = range(10000000, 100000001, 100000)

	if machine == "CPU":
		file_prefix = "vec_ops.n2_g0_c21_p42."

	operations = ["VecDot", "VecAXPY", "VecSet", "VecCopy"]
	counts = [1, 3, 3, 1]

	if machine == "CPU":
		counts[2] = 2

	# set up table
	# file = open("../plots/" + machine + "_latency_table.txt", "w+")
	file = open("../plots/" + machine + "_latency_table_CPU_clear_cache.txt", "w+")
	# file = open("../plots/" + machine + "_latency_table_42CPU.txt", "w+")
	file.write("\\begin{tabular}[b]{| l | r r | r r | r r |} \\hline \n")

	file.write("Vec size & \\multicolumn{2}{c |}{$10^3$ - $10^5$} & \\multicolumn{2}{c |}{$10^5$ - $10^7$} & \\multicolumn{2}{c |}{$10^7$ - $10^8$} \\\\ \\hline \n")
	file.write("Operation & latency & throughput & latency & throughput & latency & throughput \\\\ \\hline \n")

	for i in range(len(operations)):

		if operations[i] == "VecAXPY":
			mem_scale = 3
		elif operations[i] == "VecDot":
			mem_scale = 2
		elif operations[i] == "VecSet":
			mem_scale = 1
		elif operations[i] == "VecCopy":
			mem_scale = 2
		elif operations[i] == "VecCUDACopyTo":
			mem_scale = 1
		

		# if machine == "GPU" and operations[i] == "VecDot":
		if machine == "GPU":
			file_path = "waitforgpu-latency"
		if machine == "GPU" and operations[i] == "VecCUDACopyTo" and operations[i-1] == "VecCUDACopyTo": # pinned
			file_path = "pinned-latency"
		if machine == "CPU":
			file_path = "cpu-flush-cache-vecset-latency"

		sm_data = []
		md_data = []
		lg_data = []

		for size in sm:
			sm_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))
		for size in md:
			md_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))
		for size in lg:
			lg_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))
			if size == 100000000:
				print operations[i]
				print mem_scale*lg_data[-1]

		sm = np.array(sm)
		md = np.array(md)
		lg = np.array(lg)

		sm_data = np.array(sm_data)
		md_data = np.array(md_data)
		lg_data = np.array(lg_data)

		# prune NaNs
		mask1 = np.isfinite(sm) & np.isfinite(sm_data)
		mask2 = np.isfinite(md) & np.isfinite(md_data)
		mask3 = np.isfinite(lg) & np.isfinite(lg_data)

		s1, i1, r, p, stderr = stats.linregress(sm[mask1], sm_data[mask1])
		s2, i2, r, p, stderr = stats.linregress(md[mask2], md_data[mask2])
		s3, i3, r, p, stderr = stats.linregress(lg[mask3], lg_data[mask3]) 

		file.write(operations[i] + " & " + "{0:,.0f}".format(i1/1.0e-6) + " & " + "{0:,.0f}".format(mem_scale*1.0/(1.0e6*s1)) + " & " + "{0:,.0f}".format(i2/1.0e-6) + " & " + "{0:,.0f}".format(mem_scale*1.0/(1.0e6*s2)) + " & " + "{0:,.0f}".format(i3/1.0e-6) + " & " + "{0:,.0f}".format(mem_scale*1.0/(1.0e6*s3)) + " \\\\ \n")

	file.write("\\hline ")
	file.write("\\end{tabular}")
	file.close()

# table("CPU")


