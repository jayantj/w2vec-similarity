import numpy as np
import cProfile

def func():
  np.argpartition(np.random.uniform(size=(20000,1)),-10)
  # np.argpartition(np.random.uniform(size=(20000,15173)),-10)

def func2():
  # for i in range(20000):
    # np.argpartition(np.random.uniform(size=(1,17011)),-10)
  np.argpartition(np.random.uniform(size=(1,20000)),-10)

cProfile.run('func()')