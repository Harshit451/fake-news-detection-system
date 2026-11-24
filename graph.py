import matplotlib.pyplot as plt

# Dataset values
news_types = ["Fake News", "Real News"]
article_counts = [23481, 21417]

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Black background
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

# Both bars in blue
bars = ax.bar(
    news_types,
    article_counts,
    color="#4DA6FF",
    width=0.55
)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 400,
        f"{int(height):,}",
        ha="center",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

# Title
ax.set_title(
    "Distribution of Fake and Real News Articles",
    color="white",
    fontsize=16,
    fontweight="bold",
    pad=15
)

# Labels
ax.set_xlabel("News Type", color="white", fontsize=12)
ax.set_ylabel("Number of Articles", color="white", fontsize=12)

# Tick labels
ax.tick_params(axis="x", colors="white", labelsize=11)
ax.tick_params(axis="y", colors="white", labelsize=11)

# Grid
ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25,
    color="white"
)

# Borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("white")
ax.spines["bottom"].set_color("white")

# Y-axis limit
ax.set_ylim(0, 26000)

plt.tight_layout()

# Save for PPT
plt.savefig(
    "fake_vs_real_news_dark.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="black"
)

plt.show()