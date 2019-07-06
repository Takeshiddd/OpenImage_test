import numpy as np

def object_position_list(array, n):
    list = []
    for i in range(len(array)):
        if i % n == 0:
            list.append(array[i].tolist())
    return list

def calculate_joint_probability(distances, sigma2, dim):
    print('calculating probability')
    l2 = np.power(distances, 2)
    beki = - l2 / 2 / sigma2
    e = np.exp(beki)
    P = np.sum(e, axis=1) / (2 *sigma2 *np.pi) ** (dim / 4)
    return P

a = np.array([[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3],[1,2,3],[2,2,3],[3,2,3]], dtype = np.float32)
b = calculate_joint_probability(a, 1, 3)
print(b)