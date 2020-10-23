import pandas as pd
from facebook_scraper import get_posts

posts_df = pd.DataFrame(columns=["id", "text", "likes", "comments"])

for post in get_posts('Bristruths', pages=2500):
    if "https://bristruths.uni-truths.com/" not in post["text"]:
        text = post['text'].split("\n", 1)
        if len(text) >= 2:
            row = {"id" : text[0], "text" : text[1], "likes" : post["likes"], "comments" : post["comments"]}
            posts_df = posts_df.append(row, ignore_index=True)

posts_df.to_csv("bristruths.csv")
