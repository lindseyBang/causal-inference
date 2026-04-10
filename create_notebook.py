import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Causal Inference Workspace: DiD & Synthetic DiD

This notebook covers:
- Difference-in-Differences (DiD)
- Synthetic DiD (simplified)
- Data generation
- Estimation + plots
"""))

cells.append(nbf.v4.new_code_cell("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(42)
"""))

cells.append(nbf.v4.new_markdown_cell("## Generate Synthetic Data"))

cells.append(nbf.v4.new_code_cell("""
n_units = 50
n_periods = 20
treatment_start = 10

data = []
for u in range(n_units):
    treated = 1 if u < 10 else 0
    unit_effect = np.random.normal()
    
    for t in range(n_periods):
        time_effect = 0.5 * t
        treatment_effect = 5 if (treated and t >= treatment_start) else 0
        noise = np.random.normal()
        
        y = 10 + unit_effect + time_effect + treatment_effect + noise
        data.append([u, t, treated, y])

df = pd.DataFrame(data, columns=["unit","time","treated","outcome"])
df["post"] = (df["time"] >= treatment_start).astype(int)

df.to_csv("simulated_data.csv", index=False)
df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("## DiD Regression"))

cells.append(nbf.v4.new_code_cell("""
model = smf.ols("outcome ~ treated * post + C(unit) + C(time)", data=df).fit()
print(model.summary())
"""))

cells.append(nbf.v4.new_markdown_cell("## Plot Trends"))

cells.append(nbf.v4.new_code_cell("""
avg = df.groupby(["time","treated"])["outcome"].mean().reset_index()

for g in [0,1]:
    subset = avg[avg.treated == g]
    plt.plot(subset.time, subset.outcome, label=f"Treated={g}")

plt.axvline(x=10, linestyle="--")
plt.legend()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## Synthetic DiD (Simple Version)"))

cells.append(nbf.v4.new_code_cell("""
control = df[df.treated == 0]
treated = df[df.treated == 1]

weights = control.groupby("unit")["outcome"].mean()
weights = weights / weights.sum()

synthetic = []
for t in range(n_periods):
    val = 0
    for u, w in weights.items():
        val += w * df[(df.unit==u)&(df.time==t)]["outcome"].values[0]
    synthetic.append(val)

treated_avg = treated.groupby("time")["outcome"].mean()

plt.plot(range(n_periods), treated_avg, label="Treated")
plt.plot(range(n_periods), synthetic, label="Synthetic")
plt.axvline(x=10, linestyle="--")
plt.legend()
plt.show()
"""))

nb['cells'] = cells

with open("causal_inference_workspace.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: causal_inference_workspace.ipynb")
