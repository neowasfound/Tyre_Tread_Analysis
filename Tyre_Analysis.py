import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# 1. RESEARCH QUESTION
# Does the spray-on endurance treatment reduce
# tread loss per kilometre driven?
# 2. LOAD DATA

filepath = "C:/Users/46359990/Desktop/01_Project_Tyres_C.xlsx"
sheets = pd.read_excel(filepath, sheet_name=None)
treatments = sheets["Treatments"]

all_data = []
for sheet in sheets:
    if sheet != "Treatments":
        data = sheets[sheet].copy()
        data["id"] = sheet
        all_data.append(data)

df = pd.concat(all_data,ignore_index=True)
# Add treatment information

df = df.merge(treatments,on="id",how="left")
# 3. MISSING VALUES

tread = ["treadmeasureFL","treadmeasureFR","treadmeasureBL","treadmeasureBR"]
shocks = ["numshocksFL","numshocksFR","numshocksBL","numshocksBR"]

print("==========================================")
print("MISSING VALUE SUMMARY")
print("==========================================")

missing = df[ ["distances"] + tread + shocks].isnull().sum()
percentage = (missing / len(df)) * 100

for column in missing.index:
    print(column, ":", missing[column],"missing", "(", round(percentage[column], 2), "%)")
# Fill missing tread measurements

df[tread] = df.groupby("id")[tread].transform(lambda x: x.interpolate( method="linear" ).bfill().ffill())

# Fill missing distances with median

df["distances"] = df.groupby("id")["distances"].transform(lambda x: x.fillna(x.median()))
# Missing shocks = 0

df[shocks] = df[shocks].fillna(0)
# 4. ONE ROW PER DRIVER

results = []
for (car, treatment), group in df.groupby(["id", "treatment"]):
    group = group.sort_values("day")
    # Total distance
    total_distance = group["distances"].sum()


    # Total shocks

    total_shocks = group[shocks].sum().sum()
    # Tread loss for each tyre

    losses = []
    for column in tread:
        loss = (group[column].iloc[0]-group[column].iloc[-1]) * 1000
        losses.append(loss)

    # Average tread loss
    average_loss = np.mean(losses)
    # Wear per 1000 km
    if total_distance > 0:
        wear_per_1000 = (average_loss/ total_distance* 1000)
        shocks_per_1000 = (total_shocks/ total_distance* 1000)

    else:

        wear_per_1000 = 0
        shocks_per_1000 = 0

    results.append({

        "ID": car,
        "Treatment": treatment,
        "Total_Distance": total_distance,
        "Total_Shocks": total_shocks,
        "Shocks_Per_1000km": shocks_per_1000,
        "Avg_Wear_Loss_mm": average_loss,
        "Wear_Per_1000km": wear_per_1000
    })
summary = pd.DataFrame(results)
# 5. EXPLORE THE DATA
summary["Treatment_Label"] = summary[
    "Treatment"
].map({
    "y": "Treated",
    "n": "Control"
})

# -------------------------------
# Boxplot + Scatterplot
# -------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# Boxplot
sns.boxplot(x="Treatment_Label", y="Wear_Per_1000km", data=summary,
    palette={
        "Treated": "green",
        "Control": "red"
    },
    ax=axes[0]
)

sns.stripplot(x="Treatment_Label", y="Wear_Per_1000km", data=summary,color="black",alpha=0.6, jitter=0.2, ax=axes[0])

axes[0].set_title("Tread Wear per 1000 km by Treatment Group")
axes[0].set_ylabel("Wear Rate (mm per 1,000 km)")
axes[0].set_xlabel("Treatment Group")
# Scatterplot
sns.scatterplot( x="Total_Shocks",y="Avg_Wear_Loss_mm",hue="Treatment_Label",palette={
        "Treated": "green",
        "Control": "red"
    },
    data=summary,
    ax=axes[1],
    s=70
)
axes[1].set_title("Total Shocks vs Average Tread Loss")
axes[1].set_xlabel("Total Road Shocks")
axes[1].set_ylabel( "Average Tread Loss (mm)")
plt.tight_layout()
plt.savefig("step5_plots.png",dpi=300)
plt.show()

# -------------------------------
# Pairplot
# -------------------------------

pairplot = sns.pairplot(
    summary[
        ["Wear_Per_1000km","Total_Distance","Total_Shocks","Avg_Wear_Loss_mm","Treatment_Label"]
    ],
    hue="Treatment_Label",
    palette={
        "Treated": "#2ca02c",
        "Control": "#d62728"
    }
)

pairplot.fig.suptitle("Pairwise Relationships Across Key Features", y=1.02)
plt.savefig("pairplot.png",dpi=300,bbox_inches="tight")
plt.show()
# 6. CHECK ASSUMPTIONS

treated = summary[summary["Treatment"] == "y"]["Wear_Per_1000km"]
control = summary[ summary["Treatment"] == "n"]["Wear_Per_1000km"]

# Shapiro-Wilk for treated group
shapiro_treated = stats.shapiro(treated)
# Shapiro-Wilk for control group
shapiro_control = stats.shapiro(control)
# Create treatment code
summary["Treatment_Code"] = summary["Treatment"].map({
    "y": 1,
    "n": 0
})
# Regression variables
X = summary[["Treatment_Code","Total_Distance","Total_Shocks"]]

# Add intercept
X = sm.add_constant(X)
y = summary["Avg_Wear_Loss_mm"]
# Regression
model = sm.OLS(y,X).fit()

# Check residual normality
shapiro_residuals = stats.shapiro(model.resid)
# Treated
sm.qqplot(treated, line="s")
plt.title("Q-Q Plot: Treated Group")
plt.show()

# Control
sm.qqplot(control, line="s")
plt.title("Q-Q Plot: Control Group")
plt.show()

# Regression residuals
sm.qqplot(model.resid, line="s")
plt.title("Q-Q Plot: Regression Residuals")
plt.show()

print("\n==========================================")
print("ASSUMPTION CHECKS")
print("==========================================")
print("Treated Shapiro-Wilk:")
print("W =",round(shapiro_treated.statistic, 4),"p =",round(shapiro_treated.pvalue, 4))
print("Control Shapiro-Wilk:")
print("W =",round(shapiro_control.statistic, 4),"p =",round(shapiro_control.pvalue, 4))
print("Regression Residual Shapiro-Wilk:")
print("W =",round(shapiro_residuals.statistic, 4),"p =",round(shapiro_residuals.pvalue, 4))

# 7. STATISTICAL TESTS
# Welch t-test
t_stat, p_value = stats.ttest_ind(treated,control,equal_var=False)
print("\n==========================================")
print("STATISTICAL MODELS")
print("==========================================")
print("Welch t-test statistic:",round(t_stat, 4))
print("Welch t-test p-value:",p_value)
print("Treated mean wear:",round(treated.mean(), 4))
print("Control mean wear:",round(control.mean(), 4))
# Regression
print("\nMULTIPLE LINEAR REGRESSION")
print(model.summary())

# 8. DRIVING HABITS
print("\n==========================================")
print("DRIVING HABITS")
print("==========================================")

# Overall daily distance
print("Mean daily distance:",round(df["distances"].mean(), 2),"km")
print("Median daily distance:",round(df["distances"].median(), 2),"km")
print("Minimum daily distance:",round(df["distances"].min(), 2),"km")
print("Maximum daily distance:",round(df["distances"].max(), 2),"km")

# Compare treatment groups
comparison = summary.groupby("Treatment_Label")[["Total_Distance","Total_Shocks"]].mean()

print("\nGROUP DRIVING COMPARISON")
print(comparison.round(2))