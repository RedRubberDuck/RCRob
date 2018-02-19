from matplotlib import pyplot as plt
import numpy as np


x=np.random.randint(10,1000,300)
y=10+x*-2+np.random.random(300)*1

r=np.corrcoef(x,y)[0,1]

medianX=np.median(x)
medianY=np.median(y)
stdX=np.std(x)
stdY=np.std(y)

print(r)

RRR=medianY+r*stdY*(x-medianX)/stdX


print(np.median(RRR))
# print(RRR)




plt.figure()
plt.scatter(x,y)

plt.figure()
plt.plot(RRR)
plt.show()
