import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0, 160000, 1)
# coeff = (1 - x/160000)**2
# y = (1e-3 - 1e-5)*coeff + 1e-5
for x in [50000, 100000]:
    coeff = (1 - x/160000)**2
    y = (1e-3 - 1e-5)*coeff + 1e-5
    print(y)
# plt.plot(x, y)
# plt.savefig('lr.png')