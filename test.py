import time
import multiprocessing
const = 1

def f(x, y, z):
	time.sleep(5)
	return x.train(y, z)

class exs():
	def __init__(self,x):
		self.x = x
	def train(self, x1, x2):
		print("local train")
		return self.x+x1+x2, self.x+x1-x2


cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
exList = [exs(i) for i in range(10)]
# print(cores)
temp = pool.starmap_async(f, [(ex, 1, 2) for ex in exList]).get()
print(temp)
pool.close()
pool.join()