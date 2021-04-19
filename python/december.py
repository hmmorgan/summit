import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_floprate():

	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	vecdot_sept = []
	vecdot_dec = []
	vecaxpy_sept = []
	vecaxpy_dec = []

	for size in cpu_sizes:

		vecdot_sept.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecDot", True, 1)))
		vecdot_dec.append(float(ut.get_floprate("../data/vec-ops-december/vec_ops.n2_g0_c21_p42." + str(size) + ".795805", "VecDot", True, 1)))

		vecaxpy_sept.append(float(ut.get_floprate("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecAXPY", True, 3)))
		vecaxpy_dec.append(float(ut.get_floprate("../data/vec-ops-december/vec_ops.n2_g0_c21_p42." + str(size) + ".795805", "VecAXPY", True, 3)))
	
	# plot
	plt.plot(cpu_sizes, vecdot_sept, color="red", label="VecDot 09/19")
	plt.plot(cpu_sizes, vecdot_dec, color="red", linestyle="dashed", label="VecDot 12/19")
	plt.plot(cpu_sizes, vecaxpy_sept, color="black", label="VecAXPY 09/19")
	plt.plot(cpu_sizes, vecaxpy_dec, color="black", linestyle="dashed", label="VecAXPY 12/19")
	
	plt.title("CPU floprate 09/19-12/19", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="lower right", fontsize=12, ncol=1, frameon=False)
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()

	plt.savefig("../plots/CPU_sept_vs_dec_floprate.png")
	plt.show()

def plot_time():


	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000]#, 100000000, 1000000000]
	cpu_sizes = [10000, 100000, 1000000, 10000000]

	vecdot_sept = []
	vecdot_dec = []
	vecaxpy_sept = []
	vecaxpy_dec = []

	for size in cpu_sizes:

		vecdot_sept.append(1000000*ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecDot", 1))
		vecdot_dec.append(1000000*ut.get_time("../data/vec-ops-december/vec_ops.n2_g0_c21_p42." + str(size) + ".795805", "VecDot", 1))

		vecaxpy_sept.append(1000000*ut.get_time("../data/vec-ops/vec_ops.n2_g0_c21_p42." + str(size) + ".654910", "VecAXPY", 3))
		vecaxpy_dec.append(1000000*ut.get_time("../data/vec-ops-december/vec_ops.n2_g0_c21_p42." + str(size) + ".795805", "VecAXPY", 3))

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	plt.plot(cpu_sizes, vecdot_sept, color="red", label="VecDot 09/19")
	plt.plot(cpu_sizes, vecdot_dec, color="red", linestyle="dashed", label="VecDot 12/19")
	plt.plot(cpu_sizes, vecaxpy_sept, color="black", label="VecAXPY 09/19")
	plt.plot(cpu_sizes, vecaxpy_dec, color="black", linestyle="dashed", label="VecAXPY 12/19")
	plt.title("CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("Seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.xscale('log')
	# ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	# plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/CPU_sept_vs_dec_time.png")
	plt.show()

# plot_floprate()
plot_time()

