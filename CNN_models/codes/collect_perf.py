import sys
'''This file aims to collect performance on test data with the optimal parameter set for each TF'''
mode = sys.argv[1]

if mode == "collect_best":
    out_file = open("../out/performance_100_filters.txt", "w")
    out_file.write("\t".join(["tf", "RCmodel", "canonical", "RCaugmented", "double"]) + "\n")
    for tf in ["Scr", "Lab", "Pb", "Dfd", "Antp", "Ubx", "AbdA", "AbdB"]:
        out_file.write("Exd-" + tf + "\t")
        for mode in ["RCmodel", "canonical", "RCaugmented", "double"]:
            if mode == "RCmodel":
                perf_data = open("../out/SELEX_" + mode + "_100filters/" + tf + "/test_perf.txt", "r").readline()
            else:
                perf_data = open("../out/SELEX_" + mode + "/" + tf + "/test_perf.txt", "r").readline()
            out_file.write(perf_data.rstrip().split("\t")[1])
            if mode == "double":
                out_file.write("\n")
            else:
                out_file.write("\t")
elif mode == "overfitting":
    out_file = open("../out/performance_train_minus_test.txt", "w")
    out_file.write("\t".join(["tf", "100filters", "32filters"]) + "\n")
    for tf in ["Scr", "Lab", "Pb", "Dfd", "Antp", "Ubx", "AbdA", "AbdB"]:
        out_file.write("Exd-" + tf + "\t")
        for mode in ["RCmodel"]:
                perf_train_100 = open("../out/SELEX_" + mode + "/" + tf + "/train_perf.txt", "r").readline().rstrip().split("\t")[1]
                perf_test_100 = open("../out/SELEX_" + mode + "/" + tf + "/test_perf.txt", "r").readline().rstrip().split("\t")[1]
                out_file.write(str(float(perf_train_100)-float(perf_test_100)) + "\t")

                perf_train_32 = open("../out/SELEX_" + mode + "_32filters/" + tf + "/train_perf.txt", "r").readline().rstrip().split("\t")[1]
                perf_test_32 = open("../out/SELEX_" + mode + "_32filters/" + tf + "/test_perf.txt", "r").readline().rstrip().split("\t")[1]
                out_file.write(str(float(perf_train_32)-float(perf_test_32)) + "\n")
