#"https://medium.com/@sebastiannorena/finding-correlation-between-many-variables-multidimensional-dataset-with-python-5deb3f39ffb3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_excel('combined_data.xlsx', sheet_name='Sheet1')
data = pd.read_csv("1.csv", error_bad_lines=False)

corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns)
ax.set_yticklabels(data.columns)
plt.savefig("corr2.png")
plt.show()