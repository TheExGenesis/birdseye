# %%
from typing import Any, Callable, List, Dict, Tuple, Union

from functools import partial
import pandas as pd
import sys


if any("ipykernel" in arg for arg in sys.argv):
    from lib.prompts import (
        ONTOLOGY_GROUP_EXAMPLES,
        ONTOLOGY_GROUP_PROMPT,
        ontology,
        group_ontology,
        ONTOLOGY_LABEL_CLUSTER_PROMPT,
        ONTOLOGY_LABEL_CLUSTER_EXAMPLE,
    )
    from lib.utils import pick, retry
    from lib.openrouter_client import query_llm
else:
    from .prompts import (
        ONTOLOGY_GROUP_EXAMPLES,
        ONTOLOGY_GROUP_PROMPT,
        ontology,
        group_ontology,
        ONTOLOGY_LABEL_CLUSTER_PROMPT,
        ONTOLOGY_LABEL_CLUSTER_EXAMPLE,
    )
    from .utils import pick, retry
    from .openrouter_client import query_llm
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_label_clusters(
    tweets_df, n_top_terms=3, exclude_words=["current", "quoted", "context", "root"]
):
    if exclude_words is None:
        exclude_words = []

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tweets_df["emb_text"])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create DataFrame with TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add cluster labels to TF-IDF DataFrame
    tfidf_df["cluster"] = tweets_df["cluster"]

    # Group by cluster and sum TF-IDF scores
    cluster_tfidf = tfidf_df.groupby("cluster").sum()

    # Get top terms for each cluster, excluding specified words
    cluster_labels = {}
    for cluster, row in cluster_tfidf.iterrows():
        filtered_row = row.drop(labels=exclude_words, errors="ignore")
        top_terms = filtered_row.nlargest(n_top_terms).index.tolist()
        cluster_labels[cluster] = top_terms

    return cluster_labels


# %%


def validate_ontology_results(
    results: Dict[str, Any], ontology: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively validates results against ontology schema.

    Args:
        results: Results dict to validate
        ontology: Schema dict to validate against

    Returns:
        Dict with keys:
            valid: bool indicating if valid
            info: list of missing required keys
    """
    # print("validate results", results)

    def validate_dict(
        results_dict: Dict, schema_dict: Union[Dict, List], path: str = ""
    ) -> Dict[str, Any]:
        # print("schema_dict path", path, schema_dict)
        required_keys = [k for k in schema_dict.keys() if k != "schema_info"]
        missing = [
            f"{path}.{k}" if path else k for k in required_keys if k not in results_dict
        ]
        if missing:
            return {"valid": False, "info": missing}

        for key, value in results_dict.items():
            if key not in schema_dict or key == "schema_info":
                continue

            curr_path = f"{path}.{key}" if path else key
            schema_value = schema_dict[key]

            if isinstance(schema_value, list):
                if not isinstance(value, list):
                    return {"valid": False, "info": [f"{curr_path} should be list"]}

                schema_item = schema_value[0]
                for i, item in enumerate(value):
                    if isinstance(schema_item, dict) or isinstance(schema_item, list):
                        result = validate_dict(item, schema_item, f"{curr_path}[{i}]")
                        if not result["valid"]:
                            return result

            elif isinstance(schema_value, dict):
                if not isinstance(value, dict):
                    return {"valid": False, "info": [f"{curr_path} should be dict"]}
                result = validate_dict(value, schema_value, curr_path)
                if not result["valid"]:
                    return result

        return {"valid": True, "info": []}

    required_keys = [
        k for k in ontology.keys() if k not in ["schema_info", "low_quality_cluster"]
    ]
    missing = [k for k in required_keys if k not in results]
    if missing:
        return {"valid": False, "info": missing}

    return validate_dict(results, ontology)


def label_with_ontology(
    prompt: str,
    ontology: dict,
    return_error: bool = True,
    max_validation_retries: int = 3,
    model: str = "anthropic/claude-3.5-haiku-20241022:beta",
    **prompt_kwargs,
) -> Dict[str, Any]:
    """Labels input text using a provided ontology and prompt.

    Args:
        prompt: Template string for the prompt
        input_text: Text to be labeled
        ontology: Dictionary defining the expected structure of results
        previous_results: Optional dict of previous results to reference
        return_error: If True, returns error info instead of raising
        max_validation_retries: Number of retries on validation failure
        **prompt_kwargs: Additional kwargs to format into prompt

    Returns:
        Dict containing the labeled results matching ontology structure
    """
    # Format previous results string if any exist

    results = None
    response_text = None
    last_error = None

    for attempt in range(max_validation_retries):
        try:
            # Format prompt with all arguments
            formatted_prompt = prompt.format(
                ontology=ontology,
                **prompt_kwargs,
            )

            # Query model and extract results
            response_text = query_llm(
                message=formatted_prompt,
                model=model,
            )
            sp_toks = extract_special_tokens(response_text, tokens=["ANSWER"])
            results = parse_extracted_data(sp_toks)["ANSWER"]

            # Validate against ontology schema
            validate_ontology_results(results, ontology)
            return {**results, "is_error": False}

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            if results:
                print(results)
            elif response_text:
                print(response_text)

    # Handle all retries failed
    if not return_error:
        raise last_error

    # Return partial results with error flag if we have them
    if results:
        return {**results, "is_error": True, "error": str(last_error)}
    elif response_text:
        return {
            "is_error": True,
            "error": str(last_error),
            "response_text": response_text,
        }

    # Return error-only object if no results
    return {"is_error": True, "error": str(last_error)}


def curry(func: Callable) -> Callable:
    """Curries a function, enabling partial application of its arguments."""

    def curried_function(*args, **kwargs) -> Callable:
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        return partial(curried_function, *args, **kwargs)

    return curried_function


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
                breakpoint()
                print(f"Processing {key} failed after retries: {e}")
                results[key] = None

    if not is_dict:
        return [results[i] for i in range(len(results))]
    return results


import re
import json
from typing import List, Dict, Tuple
import openai


def extract_special_tokens(response_text: str, tokens: List[str]) -> Dict[str, str]:
    """
    Extracts JSON blocks enclosed by specified tokens from the response text.

    Args:
        response_text (str): The response text to parse.
        tokens (List[str]): A list of tokens to look for in the response.

    Returns:
        Dict[str, str]: A dictionary where the keys are tokens and the values are the extracted JSON strings.
    """
    extracted_data = {}
    for token in tokens:
        pattern = rf"<{token}>\s*({{.*?}})\s*</{token}>"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            extracted_data[token] = match.group(1)
        else:
            extracted_data[token] = None
    return extracted_data


def parse_extracted_data(extracted_data: Dict[str, str]) -> Dict[str, dict]:
    """
    Parses the extracted JSON data into Python dictionaries.

    Args:
        extracted_data (Dict[str, str]): The extracted JSON data as strings.

    Returns:
        Dict[str, dict]: A dictionary where the keys are tokens and the values are parsed JSON objects.
    """
    parsed_data = {}
    for token, json_string in extracted_data.items():
        if json_string:
            try:
                parsed_data[token] = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for token {token}: {e}")
                parsed_data[token] = None
        else:
            parsed_data[token] = None
    return parsed_data


def save_or_load_df(file_path, data_func, *args, **kwargs):
    try:
        with open(file_path, "rb") as f:
            data = pd.read_csv(f)
        print(f"Data loaded from {file_path}.")
    except Exception:
        print(f"Data not found at {file_path}. Running data function...")
        data = data_func(*args, **kwargs)
        data.to_csv(file_path, index=False)
    return data


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
import time
import random


def normalize_tweet(
    tweet_data: dict, tweet_id: str = None, child_tweet: dict = None
) -> dict:
    """Normalize tweet data to consistent format. Uses child tweet data as fallback for missing fields.

    Args:
        tweet_data: Raw tweet data from either trees or dataframe
        tweet_id: Optional tweet_id if not in tweet_data
        child_tweet: Optional child tweet to use for fallback data

    Returns:
        Normalized tweet dict with consistent property names
    """
    normalized = {}

    # ID - might be in data or passed as key
    normalized["tweet_id"] = tweet_data.get("tweet_id", tweet_id)

    # Account info - could be account_id or account_id
    normalized["account_id"] = tweet_data.get(
        "account_id", tweet_data.get("account_id")
    )
    if not normalized["account_id"] and child_tweet:
        normalized["account_id"] = child_tweet.get(
            "account_id", child_tweet.get("account_id")
        )

    # Username - fallback to reply_to_username from child
    normalized["username"] = tweet_data.get("username", "")
    if not normalized["username"] and child_tweet:
        normalized["username"] = child_tweet.get("reply_to_username", "")

    # Created at - ensure timestamp, fallback to child's created_at
    created_at = tweet_data.get("created_at")
    if created_at is None and child_tweet:
        child_created = pd.to_datetime(child_tweet.get("created_at"))
        created_at = child_created - pd.Timedelta(minutes=1)

    if isinstance(created_at, str):
        normalized["created_at"] = pd.to_datetime(created_at)
    else:
        normalized["created_at"] = created_at

    # Text content
    normalized["full_text"] = tweet_data["full_text"]

    # Engagement metrics - might be strings or ints
    normalized["retweet_count"] = int(tweet_data.get("retweet_count", 0))
    normalized["favorite_count"] = int(tweet_data.get("favorite_count", 0))

    # Reply info
    normalized["reply_to_tweet_id"] = tweet_data.get("reply_to_tweet_id")
    normalized["reply_to_user_id"] = tweet_data.get("reply_to_user_id")

    # Handle None case for reply_to_username
    reply_username = tweet_data.get("reply_to_username")
    normalized["reply_to_username"] = reply_username.lower() if reply_username else ""

    # Quote info
    normalized["quoted_tweet_id"] = tweet_data.get("quoted_tweet_id")

    # Other fields
    normalized["archive_upload_id"] = tweet_data.get("archive_upload_id")
    normalized["conversation_id"] = tweet_data.get("conversation_id")
    if not normalized["conversation_id"] and child_tweet:
        normalized["conversation_id"] = child_tweet.get("conversation_id")

    return normalized


def add_replied_tweets(
    cluster_tweets_df: pd.DataFrame,
    trees: dict,
    incomplete_trees: dict,
) -> pd.DataFrame:
    """Add replied-to tweets to a cluster's tweet dataframe using both trees and incomplete_trees.
    Uses child tweet data as fallback for missing parent tweet fields.

    Args:
        cluster_tweets_df: DataFrame containing tweets from a single cluster
        trees: Dict of complete conversation trees
        incomplete_trees: Dict of incomplete conversation trees

    Returns:
        DataFrame with replied-to tweets added, sorted by created_at
    """
    # Get IDs of tweets being replied to with their reply tweets
    reply_tweets = cluster_tweets_df[cluster_tweets_df["reply_to_tweet_id"].notna()]

    # Get replied tweets from trees and incomplete_trees
    replied_tweets = []

    # Create lookup dicts mapping tweet IDs to their containing trees
    tweet_to_tree = {
        tweet_id: tree["tweets"]
        for tree in trees.values()
        for tweet_id in tree["tweets"]
    }
    incomplete_tweet_to_tree = {
        tweet_id: tree["tweets"]
        for tree in incomplete_trees.values()
        for tweet_id in tree["tweets"]
    }

    for _, reply_tweet in reply_tweets.iterrows():
        tweet_id = reply_tweet["reply_to_tweet_id"]
        child_data = reply_tweet.to_dict()

        # Check trees first
        if tweet_id in tweet_to_tree:
            tweet_data = tweet_to_tree[tweet_id][tweet_id]
            replied_tweets.append(normalize_tweet(tweet_data, tweet_id, child_data))
            continue

        # Then check incomplete trees
        if tweet_id in incomplete_tweet_to_tree:
            tweet_data = incomplete_tweet_to_tree[tweet_id][tweet_id]
            replied_tweets.append(normalize_tweet(tweet_data, tweet_id, child_data))

    # Convert replied tweets list to DataFrame
    if replied_tweets:
        # Normalize the cluster tweets too
        cluster_tweets_normalized = [
            normalize_tweet(row.to_dict()) for _, row in cluster_tweets_df.iterrows()
        ]

        # Combine normalized data
        all_tweets = cluster_tweets_normalized + replied_tweets
        combined_df = pd.DataFrame(all_tweets)

        # Sort by date and deduplicate by tweet_id
        combined_df = combined_df.sort_values("created_at").drop_duplicates(
            subset="tweet_id", keep="first"
        )

        return combined_df

    return cluster_tweets_df


def tweet_to_str(tweet: dict, local_id: int = None) -> str:
    """Convert a tweet dict to a formatted string."""
    # Format regular tweet
    date_str = "unknown date"
    if pd.notna(tweet["created_at"]):
        if isinstance(tweet["created_at"], str):
            date_str = tweet["created_at"]
        else:
            date_str = tweet["created_at"].strftime("%d %b %Y")

    tweet_str = f'{"#" + str(local_id) if local_id else ""} @{tweet["username"] if tweet["username"] else "_"} ({date_str})\n{tweet["full_text"]}\n'
    return tweet_str


def get_cluster_tweet_texts(
    tweets_df: pd.DataFrame,
    tfidf_labels: List[str],
    max_chars: int = 50000,
    dont_include_at_top_level: List[str] = [],
) -> Tuple[str, Dict[str, int]]:
    """Get cluster summary with top TFIDF words and tweets sorted by engagement.
    Includes replied-to tweets and quoted tweets for context.
    Returns tuple of (formatted text, mapping of tweet_id to local_id)
    """
    # Start building output with TFIDF words
    output = f"Top terms: {', '.join(tfidf_labels)}\n\n"

    # Filter and sort tweets
    cluster_tweets = (
        tweets_df.copy()
        .drop_duplicates(subset=["tweet_id"], keep="first")
        .sort_values(by=["created_at"], ascending=True)
    )

    # Process all tweets first to get thread_text lengths
    displayed_tweet_ids = set()
    tweet_texts = []
    local_id_map = {}
    local_id = 1

    def dfs_replies(tweet, accum=[], depth=0):
        nonlocal local_id
        displayed_tweet_ids.add(tweet["tweet_id"])
        local_id_map[local_id] = tweet["tweet_id"]

        # Build tweet text with quote if it exists
        tweet_str = "\n".join(
            ["  " * depth + line for line in tweet_to_str(tweet, local_id).split("\n")]
        )

        # Add quoted tweet text if it exists
        if (
            "quoted_tweet_id" in tweet
            and pd.notna(tweet["quoted_tweet_id"])
            and tweet["quoted_tweet_id"] in tweets_df["tweet_id"].values
        ):
            quoted_tweet = tweets_df[
                tweets_df["tweet_id"] == tweet["quoted_tweet_id"]
            ].iloc[0]
            quoted_text = "\n".join(
                [
                    "  " * (depth + 1) + "↳ " + line
                    for line in tweet_to_str(quoted_tweet, None).split("\n")
                ]
            )
            tweet_str += "\n" + quoted_text

        if len(output + tweet_str + "\n") <= max_chars:
            accum.append(tweet_str)

        replies = tweets_df[tweets_df["reply_to_tweet_id"] == tweet["tweet_id"]]
        local_id += 1

        for _, reply in replies.iterrows():
            accum = dfs_replies(reply, accum, depth + 1)
        return accum

    cluster_tweets["thread_text"] = None
    for idx, tweet in cluster_tweets.iterrows():
        if (
            tweet["tweet_id"] not in displayed_tweet_ids
            and tweet["tweet_id"] not in dont_include_at_top_level
        ):
            thread_text_accum = []
            thread_text_accum = dfs_replies(tweet, accum=thread_text_accum)
            thread_text = "\n".join(thread_text_accum)
            cluster_tweets.loc[idx, "thread_text"] = thread_text
    # Binary search for like threshold that keeps total length under max_chars
    min_likes = 0
    max_likes = cluster_tweets["favorite_count"].max()
    while True:
        filtered_tweets = cluster_tweets[cluster_tweets["favorite_count"] >= min_likes]
        total_len = len(output) + sum(
            len(str(t)) for t in filtered_tweets["thread_text"] if pd.notna(t)
        )

        if (
            total_len <= max_chars
            or min_likes >= cluster_tweets["favorite_count"].max()
        ):
            break

        min_likes += 1
    # Use the found threshold to filter tweets
    filtered_tweets = cluster_tweets[cluster_tweets["favorite_count"] >= min_likes]
    tweet_texts = [t for t in filtered_tweets["thread_text"] if pd.notna(t)]

    # Add tweets until we hit char limit
    for tweet in tweet_texts:
        if len(output + tweet + "\n") > max_chars:
            break
        output += tweet + "\n"

    output += f"\nTop terms: {', '.join(tfidf_labels)}"
    return output, local_id_map


def make_cluster_str(
    tweet_df, trees, incomplete_trees, tfidf_labels, cluster_id, qts, max_chars=100000
):
    """Make text representation of a cluster including replied and quoted tweets.

    Args:
        tweet_df: DataFrame containing all tweets
        trees: Dict of conversation trees
        incomplete_trees: Dict of incomplete conversation trees
        tfidf_labels: Dict of TF-IDF labels for each cluster
        cluster_id: ID of cluster to process
        qts: Dict containing quote tweet data
    """
    # Get tweets for this cluster
    cluster_tweets = tweet_df[tweet_df["cluster"] == cluster_id].copy()

    # Add replied and quoted tweets
    cluster_with_replies = add_replied_tweets(cluster_tweets, trees, incomplete_trees)
    cluster_with_quotes = add_quoted_tweets(cluster_with_replies, qts)

    # Set cluster and convert IDs to strings
    cluster_with_quotes.loc[:, "cluster"] = cluster_id
    cluster_with_quotes.loc[:, "tweet_id"] = cluster_with_quotes["tweet_id"].astype(
        "str"
    )
    cluster_with_quotes.loc[:, "reply_to_tweet_id"] = cluster_with_quotes[
        "reply_to_tweet_id"
    ]

    # qts that were added and weren't in tweet_df originally
    dont_include_at_top_level = set(tweet_df.tweet_id).difference(
        set(qts["quote_map"].keys())
    )

    cluster_str, local_id_map = get_cluster_tweet_texts(
        tweets_df=cluster_with_quotes,
        tfidf_labels=tfidf_labels[cluster_id],
        max_chars=max_chars,
        dont_include_at_top_level=dont_include_at_top_level,
    )
    return cluster_str, local_id_map


def add_quoted_tweets(tweets_df: pd.DataFrame, qts: dict) -> pd.DataFrame:
    """Add quoted tweets to a dataframe using quote data from qts dict.

    Args:
        tweets_df: DataFrame containing tweets that may have quotes
        qts: Dict containing quote data with keys:
            - quoted_tweets: Dict mapping tweet_id to quoted tweet data
            - liked_quoted_tweets: Dict mapping tweet_id to liked quoted tweet data
            - quote_map: Dict mapping quoting tweet_id to quoted tweet_id

    Returns:
        DataFrame with quoted tweets added, sorted by created_at
    """
    # Get tweets that have quotes
    quoting_tweets = tweets_df[tweets_df["tweet_id"].isin(qts["quote_map"].keys())]

    # Get quoted tweets data
    quoted_tweets = []

    for _, quoting_tweet in quoting_tweets.iterrows():
        quoted_id = qts["quote_map"][quoting_tweet["tweet_id"]]

        # Check regular quoted tweets
        if quoted_id in qts["quoted_tweets"]:
            tweet_data = qts["quoted_tweets"][quoted_id]
            quoted_tweets.append(normalize_tweet(tweet_data, quoted_id))

        # Check liked quoted tweets
        # elif quoted_id in qts["liked_quoted_tweets"]:
        #     tweet_data = qts["liked_quoted_tweets"][quoted_id]
        #     quoted_tweets.append(normalize_tweet(tweet_data, quoted_id))

    # Convert quoted tweets list to DataFrame
    if quoted_tweets:
        # Normalize the original tweets too
        tweets_normalized = [
            normalize_tweet(row.to_dict()) for _, row in tweets_df.iterrows()
        ]

        # Combine normalized data
        all_tweets = tweets_normalized + quoted_tweets
        combined_df = pd.DataFrame(all_tweets)

        # Sort by date and deduplicate
        combined_df = combined_df.sort_values("created_at").drop_duplicates(
            subset="tweet_id", keep="first"
        )
        return combined_df
    return tweets_df


def group_to_string(df: pd.DataFrame, group_id: str) -> str:
    """Convert a group's clusters to a string representation."""
    clusters = df[(df["parent"] == group_id) & (df["level"] == 0)]
    return "\n\n".join(
        f"{row['name']}:\n{row['summary']}" for _, row in clusters.iterrows()
    )


def parse_response(response_text: str) -> Tuple[str, str]:
    """Parse name and summary from model response."""
    name_match = re.search(r"<NAME>(.*?)</NAME>", response_text, re.DOTALL)
    summary_match = re.search(r"<SUMMARY>(.*?)</SUMMARY>", response_text, re.DOTALL)

    if not name_match:
        raise ValueError(f"Could not parse name from response: {response_text}")
    if not summary_match:
        raise ValueError(f"Could not parse summary from response: {response_text}")

    name = name_match.group(1).strip()
    summary = summary_match.group(1).strip()

    if not name or not summary:
        raise ValueError(f"Empty name or summary parsed from response: {response_text}")

    return name, summary


def escape_json_for_template(data: dict) -> str:
    """Converts a dictionary to an escaped JSON string for use in a template."""
    json_string = json.dumps(data)  # Convert dict to JSON string
    escaped_json_string = json_string.replace('"', '\\"')  # Escape double quotes
    return escaped_json_string


def group_into_time_periods(cluster_with_replies_df, min_size=5):
    yearly_dfs = {
        year.year: df
        for year, df in cluster_with_replies_df.groupby(
            pd.Grouper(key="created_at", freq="YE")
        )
        if not df.empty
    }

    merged = {}
    pending = []

    for year, df in sorted(yearly_dfs.items()):
        # Try merging with current df
        combined_size = len(df) + sum(len(p_df) for _, p_df in pending)
        if combined_size <= min_size:
            pending.append((year, df))
            print(f"Added {year} to pending")
            continue
        # Save pending as separate group if can't merge
        pending.append((year, df))
        print(f"Merging {[x[0] for x in pending]}")
        start_year = pending[0][0]
        end_year = pending[-1][0]
        period = (
            f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
        )
        merged[period] = pd.concat([p_df for _, p_df in pending])
        pending = []

    # Handle any remaining pending groups
    if pending:
        print(f"Merging {[x[0] for x in pending]}")
        start_year = pending[0][0]
        end_year = pending[-1][0]
        period = (
            f"{start_year}" if start_year == end_year else f"{start_year}-{end_year}"
        )
        merged[period] = pd.concat([p_df for _, p_df in pending])

    return merged


def make_error_result(cluster_id: str, message: str, error: str) -> Dict[str, Any]:
    """Create an error result dict"""
    return {
        "is_error": True,
        "error": error,
        "message": message,
        "cluster_id": cluster_id,
        "cluster_summary": {"name": f"Error: {error[:50]}...", "summary": error},
        "ontology_items": {},
        "low_quality_cluster": "1",
    }


def label_one_cluster(
    cluster_id: str,
    cluster_str: str,
    model="anthropic/claude-3.5-haiku-20241022:beta",
    max_tokens: int = 8000,
    temperature: float = 0.0,
    max_retries: int = 4,
):
    """Labels a single cluster with error handling and retries"""

    print(f"labeling {cluster_id}")
    message = ONTOLOGY_LABEL_CLUSTER_PROMPT.format(
        ontology=ontology,
        tweet_texts=cluster_str,
        previous_ontology="",
        example_answer=ONTOLOGY_LABEL_CLUSTER_EXAMPLE,
    )

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            text_results = query_llm(
                message=message,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"LLM call for {cluster_id} took {time.time() - start_time:.2f}s")

        except Exception as e:
            if attempt == max_retries - 1:
                result = make_error_result(cluster_id, "", f"Query failed: {e}")
            print(f"query failed for {cluster_id}")
            continue

        try:
            answer = extract_special_tokens(text_results, tokens=["ANSWER"])
        except Exception as e:
            result = make_error_result(
                cluster_id, text_results, f"Token extraction failed: {e}"
            )
            print(f"token extraction failed for {cluster_id}")
            continue

        try:
            parsed_answer = parse_extracted_data(answer)["ANSWER"]
            if not parsed_answer:
                raise ValueError("Error parsing answer")
        except Exception as e:
            result = make_error_result(cluster_id, text_results, f"Parse failed: {e}")
            print(f"failed to parse json for {cluster_id}")
            continue
        try:
            validation_result = validate_ontology_results(parsed_answer, ontology)
        except Exception as e:
            result = make_error_result(
                cluster_id, text_results, f"Validation failed: {e}"
            )
            print(f"validation failed for {cluster_id}")
            continue
        if not validation_result["valid"]:
            result = make_error_result(
                cluster_id,
                text_results,
                f"Validation failed: {validation_result['info']}",
            )
            print(f"validation failed for {cluster_id}")
            continue
        else:
            print(f"validated {cluster_id}")
            return {
                "cluster_id": cluster_id,
                "is_error": False,
                "message": text_results,
                "ontology_items": pick(
                    [
                        k
                        for k in ontology.keys()
                        if k
                        not in ["schema_info", "low_quality_cluster", "cluster_summary"]
                    ],
                    parsed_answer,
                ),
                "cluster_summary": pick(
                    ["name", "summary"], parsed_answer["cluster_summary"]
                ),
                "low_quality_cluster": parsed_answer["low_quality_cluster"]["value"],
            }
    print(f"error result: {result}")
    return result
