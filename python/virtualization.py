import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# GPU virtualization for small vectors
def virtualization_plot(operation, count, ylims, show):

	s3 = []
	s4 = []
	s5 = []
	s6 = []
	s7 = []

	ranks = range(1, 9)

	for rank in ranks:
		s3.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(1000) + ".654911", operation, False, count)))

	for rank in ranks:
		s4.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(10000) + ".654911", operation, False, count)))

	for rank in ranks:
		s5.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(100000) + ".654911", operation, False, count)))

	for rank in ranks:
		s6.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(1000000) + ".654911", operation, False, count)))

	for rank in ranks:
		s7.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(10000000) + ".654911", operation, False, count)))

	# plot
	num = 4
	cm = plt.get_cmap('inferno')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_color_cycle([cm((1.*i)/num) for i in range(num)])

	ax.plot(ranks, s4, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="$10^4$")
	ax.plot(ranks, s5, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="$10^5$")
	ax.plot(ranks, s6, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="$10^6$")
	ax.plot(ranks, s7, marker="o", markersize="6", markeredgewidth=2, linestyle="none", label="$10^7$")

	plt.title(operation + " virtualization performance on 1 GPU", fontsize=12)
	plt.xlabel("MPI ranks", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="upper right", fontsize=12, ncol=4, frameon=False)
	plt.xlim([0.5, 8.5])
	plt.ylim(ylims)
	plt.tight_layout()
	ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

	plt.savefig("../plots/" + operation + "_virtualization_1_GPU.png")
	if show: plt.show()
	plt.gcf().clear()

def latency_table(operation, count):

	ranks = range(1, 9)
	# sizes = [1000, 10000, 100000, 1000000] # 10^3 - 10^6
	sizes = [100000, 1000000, 10000000] # 10^5 - 10^7

	# set up table
	file = open("../plots/" + operation + "_virtualization_latency_table.txt", "w+")
	file.write("\\begin{tabular}[b]{| c | r r |} \\hline \n")
	file.write("MPI ranks & latency & throughput \\\\ \\hline \n")

	# data for table
	for rank in ranks:
		data = []

		for size in sizes:
			data.append(float(ut.get_time("../data/vec-ops/vec_ops.n1_g1_c42_a" + str(rank) + "." + str(size) + ".654911", operation, count)))

		# fit regression line
		s, i, r, p, stderr = stats.linregress(np.array(sizes), data)

		file.write(str(rank) + " & " + "%.0f" %(i/1.0e-6) + " & " + "%2d" %(1.0/(1.0e6*s)) + " \\\\ \n")

	file.write("\\hline ")
	file.write("\\end{tabular}")
	file.close()

def latency_plot(operation, count):
	pass

# make plots and tables
virtualization_plot("VecDot", 1, [-5000, 110000], True)
# virtualization_plot("VecAXPY", 3, [-5000, 80000], False)
# latency_table("VecDot", 1)
# latency_table("VecAXPY", 3)



