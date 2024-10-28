# %%
import json
from datetime import datetime

with open(
    "/Users/frsc/Documents/Projects/open-birdsite-db/data/local_seed/archives/exgenesis/archive.json",
    "r",
) as f:
    archive = json.load(f)

print(f"Loaded archive with {len(archive)} keys")
# %%


def parse_twitter_date(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def patch_tweets_with_note_tweets(note_tweets, tweets):
    patched_tweets = []
    for tweet_obj in tweets:
        tweet = tweet_obj["tweet"]
        matching_note_tweet = next(
            (
                nt["noteTweet"]
                for nt in note_tweets
                if tweet["full_text"].startswith(nt["noteTweet"]["core"]["text"][:200])
                and abs(
                    (
                        parse_twitter_date(tweet["created_at"])
                        - datetime.fromisoformat(nt["noteTweet"]["createdAt"])
                    ).total_seconds()
                )
                < 1
            ),
            None,
        )

        if matching_note_tweet:
            tweet["full_text"] = matching_note_tweet["core"]["text"]

        patched_tweets.append(tweet_obj)

    return patched_tweets


patched_tweets = patch_tweets_with_note_tweets(
    archive.get("note-tweet", []), archive["tweets"]
)
print(f"Patched {len(patched_tweets)} tweets")

# %%
import pandas as pd
from tqdm import tqdm

username = archive["account"][0]["account"]["username"]
accountId = archive["account"][0]["account"]["accountId"]

print(f"Creating tweets dataframe for user {username}")


def create_tweets_df(tweets):
    print(f"Processing {len(tweets)} tweets")
    df = pd.DataFrame(
        [
            {
                "tweet_id": t["tweet"]["id_str"],
                "username": username,
                "accountId": accountId,
                "created_at": parse_twitter_date(t["tweet"]["created_at"]),
                "full_text": t["tweet"]["full_text"],
                "retweet_count": t["tweet"]["retweet_count"],
                "favorite_count": t["tweet"]["favorite_count"],
                "reply_to_tweet_id": t["tweet"].get("in_reply_to_status_id_str"),
                "reply_to_user_id": t["tweet"].get("in_reply_to_user_id_str"),
                "reply_to_username": t["tweet"].get("in_reply_to_screen_name"),
                "archive_upload_id": 1,  # assuming first upload
            }
            for t in tqdm(tweets, desc="Creating tweets dataframe")
        ]
    )
    print(f"Created dataframe with {len(df)} rows")
    return df


tweets_df = create_tweets_df(patched_tweets)
# %%
# %%
# filter df to tweets after 01-2019
tweets_df = tweets_df[tweets_df["created_at"] > pd.Timestamp("2019-01-01", tz="UTC")]
# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device="cpu",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
)


import json
import time
import numpy as np


# Function to embed all tweets and time it
def embed_tweets(tweets_df, model, batch_size=8):
    start_time = time.time()
    tweet_texts = tweets_df["full_text"].tolist()
    all_embeddings = []

    for i in range(0, len(tweet_texts), batch_size):
        batch = tweet_texts[i : i + batch_size]
        batch_embeddings = model.encode(batch)
        all_embeddings.append(batch_embeddings)

        if i % 100 == 0:  # Progress update every 100 batches
            print(f"Processed {i}/{len(tweet_texts)} tweets")

    embeddings = np.vstack(all_embeddings)  # Combine all batches
    end_time = time.time()
    print(f"Embedding all tweets took {end_time - start_time:.2f} seconds")
    return embeddings


# Call function with tweets_df
embeddings = embed_tweets(tweets_df, model, batch_size=8)

# Save embeddings to file
import pickle

with open("final_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# existing code...

# %%
queries = model.encode(["job search"])

similarities = model.similarity(queries, embeddings)

# Get top 10 indices (corrected for pandas)
top_indices = similarities[0].argsort()[-10:].flip(0).tolist()  # Convert tensor to list

# Print results
for idx in top_indices:
    print(f"\nSimilarity: {similarities[0][idx]:.3f}")
    print(f"Date: {tweets_df.iloc[int(idx)]['created_at']}")  # Ensure integer index
    print(f"Tweet: {tweets_df.iloc[int(idx)]['full_text']}")
    print("-" * 80)

# %%
