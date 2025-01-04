import math

def compute_bandwidth(n):
    b = 0
    nn = 1
    logn = []
    for j in range(len(n)):
        nn *= n[j]
        logn.append(math.log2(n[j]))

    for i in range(len(n)):
        item_to_add = 2
        for j in range(i+1):
            item_to_add*= 2*logn[j]
        
        b+=(item_to_add*math.log2(nn))

    print("bandwidth", b)
    return b

def compute_ratio(b, m, d):
    print("bandwidth old",(2**d)*math.log2(m)*math.log2(m))
    return (b/((2**d)*math.log2(m)**2))

if __name__ == '__main__':
    # v = 2

    n_list = [[7836], [5000000], [662, 524], [1015, 961], [2127, 2048], [6438, 4500], [2048, 2048], [2048, 2048], [1024, 1024], [64, 64, 64], [33, 97, 101], [128, 128, 128], [256, 256, 256]]
    m_list = [704900, 5400000, 990208, 984064, 6900600000, 88417000000, 4194304, 9.97381E+11, 9.96373E+11, 262144, 1030301, 9.35312E+17, 9.5172331E+20]


    for i in range(len(n_list)):
        print(n_list[i])
        b = compute_bandwidth(n_list[i])
        r = compute_ratio(b, m_list[i], len(n_list[i]))
        print("ratio:", r)
