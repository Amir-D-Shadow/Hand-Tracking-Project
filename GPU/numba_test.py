
from numba import jit
import numpy as np
import time

SIZE = 2000
x = np.random.random((SIZE, SIZE))


#给定n*n矩阵，对矩阵每个元素计算tanh值，然后求和。
#因为要循环矩阵中的每个元素，计算复杂度为 n*n。

def new_func(a):   # 函数在被调用时编译成机器语言

  # Numba 支持绝大多数NumPy函数
    tan_sum = 0
    for i in range(SIZE):   # Numba 支持循环
        for j in range(SIZE):
            tan_sum += np.tanh(a[i, j])
    return tan_sum

@jit
def jit_tan_sum(a):   # 函数在被调用时编译成机器语言
    tan_sum = 0
    for i in range(SIZE):   # Numba 支持循环
        for j in range(SIZE):
            tan_sum += np.tanh(a[i, j])   # Numba 支持绝大多数NumPy函数
    return tan_sum

t = time.time()
jit_tan_sum(x)
print(f"With numba jit:{time.time()-t}")

t = time.time()
new_func(x)
print(f"Without numba jit:{time.time()-t}")

"""
from numba import jit
import numpy as np
import time

SIZE = 2000
x = np.random.random((SIZE, SIZE))

@jit
def jit_tan_sum(a):   # 函数在被调用时编译成机器语言
    tan_sum = 0
    for i in range(SIZE):   # Numba 支持循环
        for j in range(SIZE):
            tan_sum += np.tanh(a[i, j])   # Numba 支持绝大多数NumPy函数
    return tan_sum

# 总时间 = 编译时间 + 运行时间
start = time.time()
jit_tan_sum(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# Numba将加速的代码缓存下来
# 总时间 = 运行时间
start = time.time()
jit_tan_sum(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
"""


