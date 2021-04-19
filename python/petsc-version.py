import utils as ut
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_floprate():

	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]

	vecdot_3_11= []
	vecdot_3_12 = []
	vecaxpy_3_11 = []
	vecaxpy_3_12 = []

	for size in cpu_sizes:

		vecdot_3_11.append(float(ut.get_floprate("../data/petsc-v3.11.3/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecDot", True, 1)))
		vecdot_3_12.append(float(ut.get_floprate("../data/petsc-v3.12/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecDot", True, 1)))

		vecaxpy_3_11.append(float(ut.get_floprate("../data/petsc-v3.11.3/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecAXPY", True, 3)))
		vecaxpy_3_12.append(float(ut.get_floprate("../data/petsc-v3.12/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecAXPY", True, 3)))
	
	# plot
	plt.plot(cpu_sizes, vecdot_3_11, color="red", label="VecDot v3.11.3")
	plt.plot(cpu_sizes, vecdot_3_12, color="red", linestyle="dashed", label="VecDot v3.12")
	plt.plot(cpu_sizes, vecaxpy_3_11, color="black", label="VecAXPY v3.11.3")
	plt.plot(cpu_sizes, vecaxpy_3_12, color="black", linestyle="dashed", label="VecAXPY v3.12")
	
	plt.title("CPU floprate", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("MFlops/second", fontsize=12)
	plt.legend(loc="lower right", fontsize=12, ncol=1, frameon=False)
	plt.xscale('log')
	plt.yscale('log')
	plt.tight_layout()

	plt.savefig("../plots/CPU_311_vs_312_floprate.png")
	plt.show()

def plot_time():


	cpu_sizes = [1000, 10000, 100000, 1000000, 10000000]#, 100000000, 1000000000]
	cpu_sizes = [10000, 100000, 1000000, 10000000]

	vecdot_3_11= []
	vecdot_3_12 = []
	vecaxpy_3_11 = []
	vecaxpy_3_12 = []

	for size in cpu_sizes:

		vecdot_3_11.append(1000000*ut.get_time("../data/petsc-v3.11.3/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecDot", 1))
		vecdot_3_12.append(1000000*ut.get_time("../data/petsc-v3.12/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecDot", 1))

		vecaxpy_3_11.append(1000000*ut.get_time("../data/petsc-v3.11.3/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecAXPY", 3))
		vecaxpy_3_12.append(1000000*ut.get_time("../data/petsc-v3.12/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecAXPY", 3))

		# vecdot_3_11.append(1000000*ut.get_time("../data/petsc-v3.11.3-fblaslapack/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecDot", 1))
		# vecdot_3_12.append(1000000*ut.get_time("../data/petsc-v3.12-fblaslapack/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecDot", 1))

		# vecaxpy_3_11.append(1000000*ut.get_time("../data/petsc-v3.11.3-fblaslapack/vec_ops.n2_g0_c21_p42.petsc_v3.11.3." + str(size), "VecAXPY", 3))
		# vecaxpy_3_12.append(1000000*ut.get_time("../data/petsc-v3.12-fblaslapack/vec_ops.n2_g0_c21_p42.petsc_v3.12." + str(size), "VecAXPY", 3))

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# plt.plot(cpu_sizes, vecdot_3_11, color="red", label="VecDot v3.11.3")
	# plt.plot(cpu_sizes, vecdot_3_12, color="red", linestyle="dashed", label="VecDot v3.12")

	# built PETSc without HYPRE in interactive session
	plt.plot(cpu_sizes, [1000000*4.9950e-06, 1000000*7.2620e-06, 1000000*2.8558e-05, 1000000*3.0591e-04], color="red", linestyle="none", marker=".", markersize=14, label="VecAXPY v3.12 without HYPRE")
	plt.plot(cpu_sizes, vecaxpy_3_12, color="black", linestyle="dashed", label="VecAXPY v3.12")
	plt.plot(cpu_sizes, vecaxpy_3_11, color="black", label="VecAXPY v3.11.3")

	plt.title("CPU execution time", fontsize=12)
	plt.xlabel("Vector size", fontsize=12)
	plt.ylabel("$10^{-6}$ seconds", fontsize=12)
	plt.legend(loc="upper left", fontsize=12, frameon=False)
	plt.xscale('log')
	# ax.ticklabel_format(axis="y", style="sci", useLocale=True)
	# plt.text(.03, 1.03, "1e-6", horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plt.tight_layout()

	plt.savefig("../plots/CPU_311_vs_312_time.png")
	# plt.savefig("../plots/CPU_311_vs_312_time_fblaslapack.png")
	plt.show()

# plot_floprate()
plot_time()