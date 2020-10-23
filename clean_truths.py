import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

truths = pd.read_csv("bristruths.csv")

truths = truths.drop("Unnamed: 0", axis=1) # fix in get_bristruths

new_text = []
for i in truths["text"]:
    text = i.replace("\n", " ")
    text = text.replace("👍", "Like:")
    text = text.replace("😢", "Sad:")
    text = text.replace("😡", "Angry:")
    text = text.replace("❤️", "Love:")
    text = text.replace("😮", "Shocked:")
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()

    if len(text) > 60:
        new_text.append(text)
    else:
        new_text.append("")

truths["text"] = new_text
truths['text'].replace('', np.nan, inplace=True)

truths = truths.dropna()

truths.to_csv("bristruths_cleaned.csv", index=False)
