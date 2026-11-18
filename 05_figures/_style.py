"""Plot style of the paper's figures (matplotlib 'seaborn' style + seaborn whitegrid)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

plt.style.use("seaborn-v0_8")  # matplotlib "seaborn" style
sns.set_style("white")
sns.set_style("whitegrid", {"grid.color": ".8", "grid.linestyle": "--"})
