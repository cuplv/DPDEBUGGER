def Alignment_kernel(X, Y=None, gamma=None):

    # python SpectralClustering_1.py --filename fop/result_time.csv --measurements no --clusters 2 --featurex size --output ./fop/result_time_spectral.csv
    
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    start = time.time()
    K = [[2.0 for x in range(len(X))] for y in range(len(X))]
    num_lines = 0
    neighbor = True
    samples = 11
    line_error = 0.2
    point_set = Set()
    point_not_set = Set()
    for t in range(len(K)):
        while True:
            i = np.random.randint(low=0,high=len(K)-1)
            if i not in point_set:
                point_set.add(i)
                point_not_set.add(t)
                break
            elif t not in point_set:
                i = t
                point_set.add(t)
                break
            else:
                if len(point_not_set)!=0:
                   i = point_not_set.pop()
                   point_set.add(i)
                   break
        for r in range(samples):
            if neighbor == False:
                j = randint(0,len(K)-1)
            else:
                coe = np.random.randint(low=2,high=10)
                while True:
                    j = np.random.randint(low=-coe*samples,high=coe*samples)
                    if i + j < len(K) and i + j > 0:
                        d = abs(X[i][0] - X[i+j][0])
                        if d > 0.0001:
                            break
                    coe = coe + 1
                j = i + j
            indecies = []
            if i < j:
                if K[i][j] == 2.0:
                    m = (X[i][1] - X[j][1])/(X[i][0] - X[j][0])
                    c = ((X[i][1] + X[j][1])/2) - (m * ((X[i][0] + X[j][0])/2))
                    num_lines += 1
                    count = 0
                    for k in range(len(K)):
                        if k != i and k != j:
                            y = m * X[k][0] + c
                            if abs(X[k][1] - y) < line_error:
                                count += 1
                                indecies.append(k)
                    if count == 0:
                        K[i][j] = sys.maxint
                        K[j][i] = sys.maxint
                    else:
                        if count > 63:
                            count = 64;
                        K[i][j] = float(1)/(2**(count))
                        K[j][i] = K[i][j]
                        for k in indecies:
                            if K[i][j] < K[i][k] and K[i][j] < K[k][j]:
                                K[i][k] = K[i][j]
                                K[k][i] = K[i][j]
                                K[j][k] = K[i][j]
                                K[k][j] = K[i][j]

            elif i == j:
                K[i][j] = 0.0
    end = time.time()
    print "kernel computation in seconds: " + str(end - start)
    print "number of fitted lines: " + str(num_lines)
    K = np.multiply(K,-gamma)
    K = np.exp(K)    # exponentiate K in-place
    return K