import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot(machine, operation, count, whole_domain, show):

	# get data
	sm = range(1000, 100001, 100)
	md = range(100000, 10000001, 10000) 
	lg = range(10000000, 100000001, 100000)

	sm_data = []
	md_data = []
	lg_data = []

	if machine == "GPU":
		# file_prefix = "vec_ops.n1_g1_c42_a1."
		file_prefix = "vec_ops.n6_g1_c7_a1."

		if operation == "VecDot":
			file_path = "waitforgpu-latency"

		# elif operation == "VecAXPY":
		else:
			file_path = "vec-ops-latency"
			# file_path = "waitforgpu-latency"
			# file_path = "gpu-flush-cache"

	if machine == "CPU":
		# file_path = "vec-ops-latency"
		file_path = "cpu-flush-cache"
		file_prefix = "vec_ops.n2_g0_c21_p42."

	# plot dots from Figure 11
	sizes = [1000, 10000, 100000, 1000000, 10000000]
	log_data = []
	log_data2 = []
	for size in sizes:

		if machine == "GPU":

			if operation == "VecDot":
				log_data.append(ut.get_time("../data/waitforgpu/vec_ops.n6_g1_c7_a1." + str(size) + ".715223", operation, count))

			elif operation == "VecAXPY":
				log_data.append(ut.get_time("../data/vec-ops/vec_ops.n6_g1_c7_a1." + str(size) + ".654914", operation, count))

		if machine == "CPU":
			log_data.append(ut.get_time("../data/cpu-flush-cache/vec_ops.n2_g0_c21_p42." + str(size), operation, count))

		log_data2.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))

	for size in sm:
		try:
			sm_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))
		except:
			sm_data.append(0)
	for size in md:
		try:	
			md_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))
		except:
			md_data.append(0)
	for size in lg:
		try:
			lg_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operation, count)))
		except:
			lg_data.append(0)

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
	else:
		plt.plot(sm, y1, color="black")
		plt.plot(md, y2, color="black")
		plt.plot(lg, y3, color="black")
	# plt.plot(sizes, log_data, marker=".", color="black", linestyle="none", markersize="8", markeredgewidth=2)
	# plt.plot(sizes, log_data2, marker=".", color="blue", linestyle="none", markersize="8", markeredgewidth=2)

	plt.title(machine + " " + operation + " execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	# plt.xscale('log')
	# plt.xlim([8e2, 1.2e7])
	# ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0))
	# ax.xaxis.set_major_locator(plt.MaxNLocator(5))

	plt.tight_layout()
	# plt.savefig("../plots/" + operation + "_" + machine + "_latency_42CPU.png") #VecDot_CPU_latency
	if show: plt.show()

def table(machine):

	# get data
	sm = range(1000, 100001, 100)
	md = range(100000, 10000001, 10000) 
	lg = range(10000000, 100000001, 100000)

	if machine == "GPU":
		# file_prefix = "vec_ops.n1_g1_c42_a1."
		file_prefix = "vec_ops.n6_g1_c7_a1."

	if machine == "CPU":
		# file_prefix = "vec_ops.n2_g0_c21_p1."
		file_prefix = "vec_ops.n2_g0_c21_p42."

	operations = ["VecDot", "VecAXPY", "VecSet", "VecCopy"]
	counts = [1, 3, 3, 1]

	if machine == "GPU":
		operations.append("VecCUDACopyTo")
		operations.append("VecCUDACopyTo") # pinned
		counts.append(1)
		counts.append(1)
	if machine == "CPU":
		counts[2] = 2

	# set up table
	# file = open("../plots/" + machine + "_latency_table.txt", "w+")
	# file = open("../plots/" + machine + "_latency_table_6GPU.txt", "w+")
	file = open("../plots/" + machine + "_latency_table_42CPU_flush_cache.txt", "w+")
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
			file_path = "cpu-flush-cache"
			# file_path = "vec-ops-latency"

		sm_data = []
		md_data = []
		lg_data = []

		for size in sm:
			sm_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))
		for size in md:
			md_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))
		for size in lg:
			lg_data.append(float(ut.get_time("../data/" + file_path + "/" + file_prefix + str(size), operations[i], counts[i])))

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

show = True
# plot("GPU", "VecDot", 1, True, show)
# plot("GPU", "VecAXPY", 3, True, show)
# plot("GPU", "VecSet", 3, True, show)
# plot("GPU", "VecCopy", 1, True, show)
plot("CPU", "VecDot", 1, True, show)
plot("CPU", "VecAXPY", 3, True, show)
plot("CPU", "VecSet", 1, True, show)
plot("CPU", "VecCopy", 1, True, show)
# table("GPU")
table("CPU")
