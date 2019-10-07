import matplotlib.pyplot as pp

# optimisation levels
opt_levels = ["-O0", "-O1", "-O2", "-O3"]
# label names for each variation
algs = ["no opt", "with opt", "parallel", "with opt+parallel"]
# short descriptions to be used as titles
descs = ["Naive algorithm with no optimisations", "Naive algorithm with nested loop tiling", "Naive unoptimised algorithm in parallel", "Naive algorithm with nested loop tiling in parallel"]

# the main data -execution times for each variation and optimisation level-
res = {alg:{lvl:{}for lvl in opt_levels} for alg in algs}

with open("output.csv", "r") as f:
    line = f.readline()
    while line:
        lvl, N, alg, elapsed = line.split(':')
        N, elapsed = int(N), int(elapsed)

        if N in res[alg][lvl]: # if N already read take average of the existing and the new
            res[alg][lvl][N] = (res[alg][lvl][N] + elapsed) // 2
        else:
            res[alg][lvl][N] = elapsed
        line = f.readline()
# assuming no errors while reading the data
X_axis = sorted(res["no opt"]["-O0"].keys())

# generating the plots for each function
for alg, desc in zip(algs, descs):
    pp.xlabel("Matrix size NxN")
    pp.ylabel("Execution time (s)")

    for lvl in opt_levels:
        sorted_vals = [res[alg][lvl][N] for N in X_axis]
        to_seconds = list(map(lambda x:float(x)/1000, sorted_vals))
        pp.plot(X_axis, to_seconds, label=lvl)

    pp.legend(loc="best")
    pp.title(desc)
    
    # save as pdf
    pp.savefig(alg.replace(' ', '_')+".pdf")
    
    # clear previous plots
    pp.clf()
    pp.cla()
    pp.close()  

# generating comparaison plot between the algorithms
for lvl in opt_levels:
    pp.yscale("log")
    pp.xlabel("Matrix size NxN")
    pp.ylabel("Execution time (ms)")

    for alg in algs:
        sorted_vals = [res[alg][lvl][N] for N in X_axis]
        pp.plot(X_axis, sorted_vals , label=alg)

    pp.title("Comaparison between algorithms with "+lvl)
    pp.legend(loc="best")
    pp.savefig("compare_%s.pdf"%(lvl[1:]))

    pp.clf()
    pp.cla()
    pp.close()  
