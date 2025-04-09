import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Simulation 1 Results.csv")
df_table1 = df[df["Run"] == 1]
plt.figure(figsize=(8, 6))
plt.plot(df_table1["Lambda"], df_table1["R_all"], marker="o", linestyle="-", color="blue")
plt.xlabel("Lambda", fontsize=12)
plt.ylabel("R_all", fontsize=12)

plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
plt.yticks([0, 0.5, 1, 1.5])

plt.title("Simulation 1 on 3-node Model with Congestion Threshold=20 and Transmission Time = 2 ", fontsize=10)
plt.grid(False)
plt.savefig("Simulation 1 table 1.png")
plt.show()
