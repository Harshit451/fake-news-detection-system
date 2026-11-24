import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Count records
fake_count = len(fake)
true_count = len(true)

# Data
labels = ["Fake News", "Real News"]
sizes = [fake_count, true_count]

# Pie chart
plt.figure(figsize=(7, 7))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    textprops={"color": "white"}
)

# Dark background
plt.gca().set_facecolor("black")
plt.gcf().set_facecolor("black")

plt.title(
    "Fake vs Real News Distribution",
    color="white",
    fontsize=16
)

plt.tight_layout()

# Save
plt.savefig(
    "fake_real_pie_chart.png",
    dpi=300,
    facecolor="white"
)

plt.show()