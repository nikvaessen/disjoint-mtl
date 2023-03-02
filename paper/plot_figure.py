import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.txt', sep='\t')
print(df)

g = sns.lineplot(df, x='layer', y='EER', hue='weights', marker="o")
g.set_title("EER on vox2 dev set")
g.set_xticks([i for i in range(0, 12)])
g.set_ylim([0, 0.5])
plt.show()
