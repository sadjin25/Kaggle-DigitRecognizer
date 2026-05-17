import matplotlib.pyplot as plt
import pandas as pd
import random

df = pd.read_csv("./Datas/train.csv")

NUMS = 25

plt.figure(figsize=(NUMS,4))

for i in range(NUMS):
    randIdx = random.randint(0, len(df)-1)
    label = df.iloc[randIdx, 0]
    img = df.iloc[randIdx, 1:].to_numpy().reshape(28,28)

    plt.subplot(5, 5, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"label : {label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
