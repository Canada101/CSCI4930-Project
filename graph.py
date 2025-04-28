import pandas as pd
import matplotlib.pyplot as plt

def plot_simulation_results(
    csv_file: str,
    run_id: int = None,
    x_col: str = "Lambda",
    y_col: str = "R_all",
    xticks: list = None,
    yticks: list = None,
    title: str = None,
    output_file: str = None
):
    """
    Plot simulation results from a CSV. 
    If 'Run' column is present and run_id is not None, filters by that run.
    Otherwise plots all rows.
    """
    df = pd.read_csv(csv_file)

    if run_id is not None and "Run" in df.columns:
        df = df[df["Run"] == run_id]

    plt.figure(figsize=(8, 6))
    plt.plot(df[x_col], df[y_col], marker="o", linestyle="-")
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.grid(False)

    if output_file:
        plt.savefig(output_file)
    plt.show()

plot_simulation_results(
    csv_file="table11_sim3.csv",
    run_id=None,             
    x_col="h",              
    y_col="R_all",           
    xticks=[0.02,0.05,0.07,0.10,0.12,0.13,0.14],
    yticks=[0.9880,0.9873, 0.9881,0.9882,0.9874,0.9881,0.9873],
    title="Simulation3(Flood Routing) 5-Node Model Two",
    output_file="sim3_5nodeModelTwo.png"
)
