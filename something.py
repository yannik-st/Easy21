import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

a = np.array([0,1,0])
b = np.array([0,0,0,1,1,0])
c = np.array([0,1])

'''ab = np.outer(a,b)

print('ab:')
print(ab)

abc = np.reshape(np.outer(ab,c), newshape=36)
print('abc:')
print(abc)
print(np.shape(abc))'''
h = np.array([0,0,1,0,0,0,0,0,1])
print(h)
print(h.reshape(-1,1))