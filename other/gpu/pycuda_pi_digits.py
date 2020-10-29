import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import time

kernels = SourceModule("""
__global__ void custom_kernel(int *g_y, int *g_x)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int x = g_x[i];
  atomicAdd(g_y+x, 1);
}
""")

custom_kernel = kernels.get_function("custom_kernel");
size = 5120000
block_size = 512 # design a 1d block and grid structure
grid_size = int(size/block_size)
block = (block_size,1,1)
grid = (grid_size,1)

print("Creating a big dataset")
dataset = ""
for i in range(1, 9):
   dataset = dataset + open(f'pi200m.ascii.0{str(i)}of20', 'r').read()
print("Done creating a big dataset")

numbers = dataset.replace(" ", "")
numbers = numbers.replace("\n", "")
X = numpy.array([int(s) for s in numbers])
print(time.strftime('%H:%M:%S'), 'Calc gpu freqs')
X_gpu = gpuarray.to_gpu(X)
Y_gpu = gpuarray.zeros(10, int) # 1. transfer to GPU
custom_kernel(Y_gpu, X_gpu, block=block, grid=grid) # 2. execute kernel
print(Y_gpu)
print(time.strftime('%H:%M:%S'), 'Done')

"""
C:\Anaconda\python.exe "C:/Work/Deep learning/GPU/pycuda_digits.py"
Creating a big dataset
Done creating a big dataset
16:40:02 Calc gpu freqs
[511691 511954 511556 511875 512486 511939 510831 512877 511920 512871]
16:40:05 Done
"""