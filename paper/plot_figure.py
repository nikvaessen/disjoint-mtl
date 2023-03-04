import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.txt", sep="\t")
print(df)
df["EER"] = df["EER"] * 100
sns.set_theme()

sns.set(rc={"figure.figsize": (11.7 / 2, 3.2)})
hue_order = ["pre-trained", "STL", "MTL joint", "MTL dj 2s", "MTL dj 10s"]
g = sns.lineplot(
    df,
    x="layer",
    y="EER",
    style="weights",
    hue="weights",
    hue_order=hue_order,
    markers=True,
    dashes=False
)
g.set_xticks([i for i in range(0, 12)])
g.set_ylim([0, 50])
g.set(xlabel="Layer from which speaker embedding is computed", ylabel="EER in %")
g.set_title("EER on development subset of VoxCeleb2")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
# plt.show()

plt.tight_layout()
plt.savefig("layer_vs_eer.pdf")
