# %%
import logging
import pickle
import os
import sys
import re
from collections import defaultdict, deque
from datetime import datetime


from collections import defaultdict, deque


from tqdm import tqdm

import pandas as pd
from tqdm import tqdm
import re

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List, Callable, Any, Union
import time
import random


def parallel_io_with_retry(
    func: Callable,
    data: Union[List[Any], Dict[Any, Any]],
    max_workers: int = 5,
    max_retries: int = 3,
    delay: int = 2,
) -> Union[List[Any], Dict[Any, Any]]:
    """
    Runs an I/O-bound function with retries and parallel execution.

    Args:
        func (Callable): The I/O-bound function to be run. Should take one argument from `data`.
        data (Union[List[Any], Dict[Any, Any]]): The data to be passed to the function.
        max_workers (int): The number of workers for the ThreadPoolExecutor.
        max_retries (int): The maximum number of retries for each function call.
        delay (int): The delay between retries in seconds.

    Returns:
        Union[List[Any], Dict[Any, Any]]: A list or dict of results.
    """
    is_dict = isinstance(data, dict)
    if not is_dict:
        data = {i: item for i, item in enumerate(data)}

    results = {}

    # Decorate the function with retry logic
    func_with_retry = retry(max_retries=max_retries, delay=delay)(func)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(func_with_retry, item): key for key, item in data.items()
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"Processing {data[key]} failed after retries: {e}")
                results[key] = None

    if not is_dict:
        return [results[i] for i in range(len(results))]
    return results


def parse_twitter_date(date_str):
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def parse_iso_date(date_str):
    # Remove the 'Z' and add UTC timezone
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str)


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
                        - parse_iso_date(nt["noteTweet"]["createdAt"])
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


def create_tweets_df(tweets, username, accountId):
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


def clean_tweet_text(text):
    # Remove links of the form "https://t.co/{id}"
    text = re.sub(r"https://t\.co/\w+", "", text)

    # Remove retweet prefix "RT @username:"
    text = re.sub(r"^RT @[A-Za-z0-9_]+: ", "", text)

    # Remove "@" mentions and extra whitespace at the beginning
    text = re.sub(r"^(\s*@\w+\s+)+", "", text)

    return text.strip()  # Remove leading/trailing whitespace


def get_liked_tweets(supabase, tweet_ids):
    """
    Get liked tweets that exist for the given tweet IDs.
    Returns dict of tweet_id -> full_text
    """
    response = (
        supabase.table("liked_tweets")
        .select("tweet_id, full_text")
        .in_("tweet_id", tweet_ids)
        .execute()
    )

    print(f"Found {len(response.data)} of {len(tweet_ids)} liked tweets")

    return {row["tweet_id"]: {"full_text": row["full_text"]} for row in response.data}


def get_tweets(supabase, tweet_ids):
    """
    Get tweets that exist for the given tweet IDs.
    Returns dict of tweet_id -> tweet data matching 04_tweets.sql schema
    """
    response = supabase.table("tweets").select("*").in_("tweet_id", tweet_ids).execute()

    print(f"Found {len(response.data)} of {len(tweet_ids)} tweets")

    return {
        row["tweet_id"]: {
            "tweet_id": row["tweet_id"],
            "account_id": row["account_id"],
            "created_at": row["created_at"],
            "full_text": row["full_text"],
            "retweet_count": row["retweet_count"],
            "favorite_count": row["favorite_count"],
            "reply_to_tweet_id": row["reply_to_tweet_id"],
            "reply_to_user_id": row["reply_to_user_id"],
            "reply_to_username": row["reply_to_username"],
            "archive_upload_id": row["archive_upload_id"],
        }
        for row in response.data
    }


def get_conversation_ids(supabase, tweet_ids):
    """
    Get conversation IDs for the given tweet IDs.
    Returns dict of tweet_id -> conversation_id
    """
    response = (
        supabase.table("conversations")
        .select("tweet_id, conversation_id")
        .in_("tweet_id", tweet_ids)
        .execute()
    )

    print(f"Found {len(response.data)} of {len(tweet_ids)} conversation IDs")

    return {row["tweet_id"]: row["conversation_id"] for row in response.data}


def get_tweets_by_conversation_ids(supabase, conversation_ids):
    """
    Get all tweets that belong to the given conversation IDs.
    Returns list of tweets ordered by created_at.
    """
    # Filter out None/null conversation_ids
    valid_conv_ids = [cid for cid in conversation_ids if cid]

    if not valid_conv_ids:
        return []

    response = (
        supabase.table("tweets_w_conversation_id")
        .select(
            """
            tweet_id,
            created_at,
            full_text,
            reply_to_tweet_id,
            favorite_count,
            retweet_count
        """
        )
        .in_("conversation_id", valid_conv_ids)
        .order("created_at")
        .execute()
    )

    print(
        f"Retrieved {len(response.data)} tweets from {len(valid_conv_ids)} conversation IDs"
    )

    return response.data


def build_conversation_trees(tweets):
    """
    Organize tweets into conversation trees.
    Returns dict of conversation_id -> {
        'root': tweet_id of root,
        'tweets': dict of tweet_id -> tweet data,
        'children': dict of tweet_id -> list of child tweet_ids,
        'parents': dict of tweet_id -> parent tweet_id,
        'paths': dict of leaf_id -> list of tweet_ids from root to leaf
    }
    """
    conversations = {}

    # Organize tweets by conversation
    for tweet in tqdm(tweets, desc="Building conversations"):
        conv_id = tweet["conversation_id"]
        if conv_id not in conversations:
            conversations[conv_id] = {
                "tweets": {},
                "children": defaultdict(list),
                "parents": {},
                "root": None,
                "paths": {},
            }

        tweet_id = tweet["tweet_id"]
        conversations[conv_id]["tweets"][tweet_id] = tweet

        reply_to = tweet.get("reply_to_tweet_id")
        if reply_to:
            conversations[conv_id]["children"][reply_to].append(tweet_id)
            conversations[conv_id]["parents"][tweet_id] = reply_to
        else:
            conversations[conv_id]["root"] = tweet_id

    # Build paths iteratively
    for conv in tqdm(conversations.values(), desc="Building paths"):
        root = conv["root"]
        if not root:
            continue

        visited = set()  # Track visited tweet IDs
        stack = [(root, [root])]

        while stack:
            current_id, path = stack.pop()
            children = conv["children"].get(current_id, [])

            if not children:
                conv["paths"][current_id] = path
            else:
                # Only process unvisited children
                unvisited = [c for c in children if c not in visited]
                if unvisited:
                    # print(f"Adding {len(unvisited)} unvisited children of {current_id} to stack")
                    for child_id in unvisited:
                        visited.add(child_id)
                        stack.append((child_id, path + [child_id]))

    return conversations


def get_incomplete_reply_chains(tweets_df, supabase, batch_size=100):
    """
    Go up the reply_to_tweet_id chain of tweets that are replies but don't have conversation IDs.
    Also checks liked_tweets table.

    Works in batches of batch_size.
    Returns dict of tweet_id -> tweet OR liked_tweet
    """
    incomplete_replies = tweets_df[
        tweets_df["reply_to_tweet_id"].notna() & tweets_df["conversation_id"].isna()
    ]

    current_ids = set(tweets_df.tweet_id.to_list())
    incomplete_reply_ids = incomplete_replies["reply_to_tweet_id"].to_list()

    tweet_queue = deque(incomplete_reply_ids)
    found_tweets = {}
    tweet_checked = set()

    liked_queue = deque()
    found_liked = {}

    while tweet_queue:
        batch = list(tweet_queue)[:batch_size]
        for _ in range(min(batch_size, len(tweet_queue))):
            tweet_queue.popleft()

        new_tweets = get_tweets(supabase, batch)
        found_tweets.update(new_tweets)
        # add the reply_to_tweet_ids of the new tweets to queue, as long as they're not in found_tweets or found_liked
        tweet_queue.extend(
            tid
            for tid in new_tweets.keys()
            if tid not in found_tweets
            and tid not in found_liked
            and tid not in tweet_checked
        )
        liked_queue.extend(tid for tid in batch if tid not in found_tweets)
        tweet_checked.update(batch)

    # deduplicate liked_queue
    liked_queue = deque(set(liked_queue).difference(current_ids))

    while liked_queue:
        batch = list(liked_queue)[:batch_size]
        for _ in range(min(batch_size, len(liked_queue))):
            liked_queue.popleft()

        new_liked = get_liked_tweets(supabase, batch)
        found_liked.update(new_liked)

    return found_tweets, found_liked


def df_to_tweet_dict(df):
    """Convert DataFrame to dict of tweet dicts keyed by tweet_id.
    Handles duplicate tweet_ids by keeping the first occurrence."""
    # Drop duplicates keeping first occurrence
    df_unique = df.drop_duplicates(subset="tweet_id", keep="first")
    return df_unique.set_index("tweet_id").to_dict("index")


def build_incomplete_conversation_trees(found_tweets, found_liked):
    """
    Build conversation trees from incomplete reply chains.

    Args:
        found_tweets: dict of tweet_id -> tweet data
        found_liked: dict of tweet_id -> liked tweet data

    Returns:
        Dict of root_id -> {
            'root': root_id,
            'tweets': dict of tweet_id -> tweet data,
            'children': dict of tweet_id -> list of child ids,
            'parents': dict of tweet_id -> parent id,
            'paths': dict of leaf_id -> list of tweet_ids from root to leaf
        }
    """
    # Combine tweets and build parent relationships
    all_tweets = {**found_tweets, **found_liked}
    parents = {}
    children = defaultdict(list)

    # Build parent/child relationships
    for tweet_id, tweet in found_tweets.items():
        reply_to = tweet.get("reply_to_tweet_id")
        if reply_to and reply_to in all_tweets:
            parents[tweet_id] = reply_to
            children[reply_to].append(tweet_id)

    # Find roots (tweets with no parents)
    roots = {tid: tweet for tid, tweet in all_tweets.items() if tid not in parents}

    trees = {}
    # Build tree for each root
    for root_id in roots:
        tree = {
            "root": root_id,
            "tweets": {root_id: all_tweets[root_id]},
            "children": defaultdict(list),
            "parents": {},
            "paths": {},
        }

        # BFS to build paths
        queue = deque([(root_id, [root_id])])
        while queue:
            current_id, path = queue.popleft()

            if current_id in children:
                tree["children"][current_id].extend(children[current_id])
                for child_id in children[current_id]:
                    tree["parents"][child_id] = current_id
                    tree["tweets"][child_id] = all_tweets[child_id]
                    queue.append((child_id, path + [child_id]))
            else:
                # Leaf node - store path
                tree["paths"][current_id] = path

        trees[root_id] = tree

    return trees


def get_thread_embedding_text(tweet_id, trees, max_chars=1024, include_date=False):
    """
    Create embedding text for a tweet with its conversation context,
    assembled in chronological order.

    Args:
        tweet_id: ID of the tweet
        trees: Conversation trees dict
        max_chars: Maximum characters in output text
        include_date: Whether to include tweet dates
    """
    # Find conversation
    conv = next((tree for tree in trees.values() if tweet_id in tree["tweets"]), None)
    if not conv:
        return None

    tweet = conv["tweets"][tweet_id]
    parts = []

    def format_tweet(tweet_type, tweet_data):
        text = clean_tweet_text(tweet_data["full_text"])
        if include_date and "created_at" in tweet_data:
            date = parse_twitter_date(tweet_data["created_at"]).strftime("%d %b %Y")
            return f"[{tweet_type} {date}] {text}"
        return f"[{tweet_type}] {text}"

    # Collect parts in chronological order
    root_id = conv["root"]
    if root_id and root_id != tweet_id:
        parts.append(format_tweet("root", conv["tweets"][root_id]))

    current_id = conv["parents"].get(tweet_id)
    context_parts = []
    while current_id and current_id != root_id:
        context_parts.append(format_tweet("context", conv["tweets"][current_id]))
        current_id = conv["parents"].get(current_id)
    parts.extend(reversed(context_parts))

    # Add current tweet
    parts.append(format_tweet("current", tweet))

    # select tweet first
    # Assemble within max_chars
    selected, chars = [], 0

    # Define the order to check parts
    check_order = []
    i, j = 0, len(parts) - 1
    if len(parts) == 1:
        check_order.append(parts[0])
    if len(parts) == 2:
        check_order.append(parts[1])
        check_order.append(parts[0])
    if len(parts) == 3:
        check_order.append(parts[2])
        check_order.append(parts[1])
        check_order.append(parts[0])
    if len(parts) > 3:
        check_order.append(parts[-1])
        check_order.append(parts[-2])
        check_order.append(parts[0])
        check_order.extend(reversed(parts[1:-2]))

    # Select parts without exceeding max_chars
    selected_set = set()
    chars = 0
    for part in check_order:
        part_len = len(part) + 1
        if chars + part_len > max_chars:
            break
        selected_set.add(part)
        chars += part_len

    # Assemble selected parts in original order
    selected = []
    for part in parts:
        if part in selected_set:
            selected.append(part)
        else:
            if "<...>" not in selected:
                selected.append("<...>")
    return "\n".join(selected)


def retry(max_retries=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    time.sleep(delay)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_all_conversation_ids(supabase, reply_ids):
    """Get conversation IDs for all reply tweets in parallel batches"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create batches of 100 IDs
    batches = [reply_ids[i : i + 100] for i in range(0, len(reply_ids), 100)]

    def get_batch_conv_ids(batch):
        return get_conversation_ids(supabase, batch)

    # Process batches in parallel
    results = parallel_io_with_retry(
        get_batch_conv_ids, batches, max_workers=5, max_retries=5, delay=3
    )

    # Combine results
    conv_map = {}
    for batch_result in results:
        if batch_result:
            conv_map.update(batch_result)

    logger.info(f"Found {len(conv_map)} of {len(reply_ids)} conversation IDs")
    return conv_map


def get_all_conversation_tweets(supabase, conv_map):
    """Get tweets for all conversations in parallel batches"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Create batches of 100 conversation IDs
    conv_values = list(conv_map.values())
    batches = [conv_values[i : i + 100] for i in range(0, len(conv_values), 100)]

    def get_batch_tweets(batch):
        return get_tweets_by_conversation_ids(supabase, batch)

    # Process batches in parallel
    results = parallel_io_with_retry(
        get_batch_tweets, batches, max_workers=5, max_retries=5, delay=3
    )

    # Combine results
    conversation_tweets = []
    for batch_tweets in results:
        if batch_tweets:
            conversation_tweets.extend(batch_tweets)

    logger.info(
        f"Retrieved {len(conversation_tweets)} tweets from {len(conv_map)} conversation IDs"
    )
    return conversation_tweets


def process_tweets(supabase, archive, username, include_date=False):
    accountId = archive["account"][0]["account"]["accountId"]
    patched_tweets = patch_tweets_with_note_tweets(
        archive.get("note-tweet", []), archive["tweets"]
    )
    tweets_df = create_tweets_df(patched_tweets, username, accountId)

    # Ensure IDs are strings
    tweets_df["tweet_id"] = tweets_df["tweet_id"].astype(str)
    tweets_df["accountId"] = tweets_df["accountId"].astype(str)
    tweets_df["reply_to_tweet_id"] = tweets_df["reply_to_tweet_id"].astype(str)

    # filter df to tweets after 01-2019 if needed
    if username == "exgenesis":
        tweets_df = tweets_df[
            tweets_df["created_at"] > pd.Timestamp("2019-01-01", tz="UTC")
        ]
    reply_ids = tweets_df[tweets_df.reply_to_tweet_id.notna()].tweet_id.to_list()
    conv_map = get_all_conversation_ids(supabase, reply_ids)
    tweets_df["conversation_id"] = tweets_df["reply_to_tweet_id"].map(conv_map)

    conversation_tweets = get_all_conversation_tweets(supabase, conv_map)
    trees = build_conversation_trees(conversation_tweets)

    tweets_df["emb_text"] = tweets_df["full_text"].apply(
        lambda x: f"[current] {clean_tweet_text(x)}"
    )

    n_replies_w_conv_id = tweets_df["conversation_id"].notna().sum()
    n_replies = tweets_df["reply_to_tweet_id"].notna().sum()
    print(f"Found {n_replies_w_conv_id}/{n_replies} replies with conversation IDs")

    replies_w_no_conv_id = tweets_df[
        tweets_df["reply_to_tweet_id"].notna() & tweets_df["conversation_id"].isna()
    ]

    found_tweets, found_liked = get_incomplete_reply_chains(
        replies_w_no_conv_id, supabase, batch_size=100
    )

    found_and_old = {**found_tweets, **df_to_tweet_dict(replies_w_no_conv_id)}
    incomplete_trees = build_incomplete_conversation_trees(found_and_old, found_liked)

    tweets_df.loc[tweets_df["conversation_id"].notna(), "emb_text"] = tweets_df.loc[
        tweets_df["conversation_id"].notna(), "tweet_id"
    ].apply(lambda x: get_thread_embedding_text(x, trees, include_date=include_date))

    tweets_df.loc[
        tweets_df["reply_to_tweet_id"].notna() & tweets_df["conversation_id"].isna(),
        "emb_text",
    ] = tweets_df.loc[
        tweets_df["reply_to_tweet_id"].notna() & tweets_df["conversation_id"].isna(),
        "tweet_id",
    ].apply(
        lambda x: get_thread_embedding_text(
            x, incomplete_trees, include_date=include_date
        )
    )

    tweets_df = tweets_df[tweets_df["emb_text"] != ""]
    print(f"Filtered to {len(tweets_df)} tweets")
    return tweets_df.reset_index(drop=True), trees, incomplete_trees
