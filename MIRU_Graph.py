import csv
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
sns.set_style('whitegrid', {'grid.linestyle': '--','xtick.color': '.15','axes.edgecolor': '.15', 'xtick.bottom': True, 'ytick.left': True})

open_file_path = './only/train_only_Word_pare_P_sigma1_sorted.csv'
# open_file_path = 'hikaku/bus-train_beki_sabun.csv'
y = []
with open(open_file_path, 'r') as f:
        reader = csv.reader(f)
        i = 1545
        for row in reader:
          
            y.append([i, np.exp(float(row[2]))])
            i -= 1
# x = range(0,len(y))
plt.figure(1)
# plt.bar(x, y[::-1], width = 0.9, color = 'green')
# plt.yscale('log')
# plt.xlabel('x')  # x軸のラベル
# plt.ylabel("P")  # y軸のラベル
# plt.show()

# sir_x = pd.Series(x, index=x)
# sir_y = pd.Series(y, index=x)
df1 = pd.DataFrame(y, columns=['Rank of words', 'P'])

sns.lineplot(x = 'Rank of words', y = 'P', data = df1)
plt.yscale('log')
plt.xticks(np.array([1, 500, 1000, 1500]))
plt.yticks(np.array([10**-240, 10**-220, 10**-200, 10**-180, 10**-160]))
# plt.yticks(np.array([10**-40, 10**-20, 1, 10**20, 10**40]))
plt.xlabel('Rank of words')  # x軸のラベル
plt.ylabel("P")  # y軸のラベル
# plt.ylabel("P'/ P")  # y軸のラベル
plt.show()