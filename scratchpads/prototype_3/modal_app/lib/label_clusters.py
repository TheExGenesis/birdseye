from typing import Any, Callable, List, Dict, Tuple, Union
from functools import partial
import pandas as pd


LABEL_CLUSTER_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following cluster of tweets:

<TWEETS>
{tweet_texts}
</TWEETS>

The name should be straightforward and reflect the overall theme or topic of the tweets, as the user would write it. Begin the name with a representative emoji.
The summary should be concise and capture common themes, in the style of the user. Don't quote tweets directly.
If you can't find a name or summary, set the name to "No name found" and the summary to "No summary found".

Respond in JSON format, delimited by <ANSWER>:
<ANSWER>
{{
    "name": "📕 A punchy handle for this cluster",
    "summary": "A summary of the cluster themes",
    "bad_group_flag": 0 or 1. Set to 1 if you think this grouping is ambiguous or low signal - stuff like small talk, short reactions, urls, etc."
}}
</ANSWER>
"""

LABEL_GROUP_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following group of related clusters:

<CLUSTERS>
{group_str}
</CLUSTERS>

The name should be straightforward and reflect the overall theme or topic of the tweets, as the user would write it. Begin the name with a representative emoji.
The summary should be concise and capture common themes, in the style of the user. Don't quote tweets directly.
If you can't find a name or summary, set the name to "No name found" and the summary to "No summary found".



Respond in JSON format, delimited by <ANSWER>:
<ANSWER>
{{
    "name": "📚 A title for this group of clusters",
    "summary": "A summary of how these clusters relate",
    "bad_group_flag": 0 or 1. Set to 1 if you think this grouping is ambiguous, low signal, or doesn't make sense."
}}
</ANSWER>
"""


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
                print(f"Processing {data[key]} failed after retries: {e}")
                results[key] = None

    if not is_dict:
        return [results[i] for i in range(len(results))]
    return results


import re
import json
from typing import List, Dict, Tuple
import anthropic


def query_anthropic_model(
    prompt: str,
    model: str = "claude-3-5-haiku-20241022",
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    """
    Sends a prompt to the Anthropic model and returns the response text.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The Anthropic model to use.
        max_tokens (int): The maximum number of tokens for the response.
        temperature (float): The temperature setting for randomness.

    Returns:
        str: The response text from the model.
    """
    try:
        client = anthropic.Anthropic()
        send_msg = {"role": "user", "content": prompt}
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[send_msg],
        )
        text = response.content[0].text
        print(f"Anthropic response: {text}")
        return text
    except anthropic.APIError as e:
        print(f"Anthropic API error: {e}")
        raise
    except anthropic.APIConnectionError as e:
        print(f"Connection error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


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


import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
import time
import random


def get_cluster_tweet_texts(tweets_df: pd.DataFrame, cluster_id: int) -> str:
    """Get the text of the top tweets in a cluster."""
    # Create mask first
    non_retweet_mask = ~tweets_df["full_text"].str.startswith("RT @")
    cluster_mask = tweets_df["cluster"] == cluster_id

    # Apply both masks in single operation
    cluster_tweets_df = (
        tweets_df[non_retweet_mask & cluster_mask]
        .sort_values(
            by=["cluster_prob", "favorite_count", "retweet_count"], ascending=False
        )
        .head(20)
    )

    return "\n".join(cluster_tweets_df["full_text"].tolist())


def label_cluster(cluster_id: int, tweet_texts: str) -> Dict[str, Any]:
    """Processes a single cluster and returns the parsed results dict."""
    try:
        response_text = query_anthropic_model(
            LABEL_CLUSTER_PROMPT.format(tweet_texts=tweet_texts)
        )
        sp_toks = extract_special_tokens(response_text, tokens=["ANSWER"])
        results = parse_extracted_data(sp_toks)["ANSWER"]

        if not results or not all(k in results for k in ["name", "summary"]):
            raise ValueError(f"Invalid response format: {response_text}")

        print(f"Processed cluster {cluster_id}: {results['name']}")
        print(results)
        return {**results, "cluster_id": cluster_id}

    except Exception as e:
        print(f"Error processing cluster {cluster_id}: {e}")
        raise e


def label_all_clusters(
    tweets_df: pd.DataFrame,
    max_workers: int = 5,
    max_retries: int = 5,
    delay: int = 2,
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Processes all clusters concurrently using parallel_io_with_retry.
    Returns dict with cluster results.
    """
    # Filter out -1 cluster and create input dict
    valid_clusters = tweets_df.cluster.unique()
    cluster_dict = {
        cluster_id: cluster_id for cluster_id in valid_clusters if cluster_id != "-1"
    }

    def label_cluster_with_inputs(cluster_id: int) -> Dict[str, Any]:
        return label_cluster(cluster_id, tweets_df)

    results = parallel_io_with_retry(
        func=label_cluster_with_inputs,
        data=cluster_dict,
        max_workers=max_workers,
        max_retries=max_retries,
        delay=delay,
    )

    return results


def group_to_string(df: pd.DataFrame, group_id: str) -> str:
    """Convert a group's clusters to a string representation."""
    clusters = df[(df["parent"] == group_id) & (df["level"] == 0)]
    return "\n\n".join(
        f"{row['name']}:\n{row['summary']}" for _, row in clusters.iterrows()
    )


def label_cluster_groups(
    hierarchy_df: pd.DataFrame,
    max_workers: int = 5,
    max_retries: int = 3,
    delay: int = 2,
) -> pd.DataFrame:
    """Label groups of clusters using the API in parallel."""
    groups = [
        g
        for g in hierarchy_df[hierarchy_df.level == 1]["cluster_id"].unique()
        if g != "-1"
    ]
    # Prepare the input data for parallel processing
    group_strings = {
        group_id: group_to_string(hierarchy_df, group_id) for group_id in groups
    }

    # Process groups in parallel
    results = parallel_io_with_retry(
        func=label_single_group,
        data=group_strings,
        max_workers=max_workers,
        max_retries=max_retries,
        delay=delay,
    )

    # Convert results to DataFrame format
    processed_results = []
    for group_id, result in results.items():
        if result:
            result["cluster_id"] = group_id  # Use cluster_id to match schema
        else:
            result = {
                "cluster_id": group_id,
                "name": f"Error Processing Group {group_id}",
                "summary": "Error occurred during processing",
                "bad_group_flag": "1",
            }
        processed_results.append(result)

    return pd.DataFrame(processed_results)


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


def label_single_group(group_str: str) -> Dict[str, str]:
    """Process a single group of clusters using the Anthropic API.

    Args:
        group_str (str): String representation of the group's clusters

    Returns:
        Dict[str, str]: Dictionary containing name, summary, and bad_group_flag
    """

    try:
        response_text = query_anthropic_model(
            LABEL_GROUP_PROMPT.format(group_str=group_str)
        )
        sp_toks = extract_special_tokens(response_text, tokens=["ANSWER"])
        results = parse_extracted_data(sp_toks)["ANSWER"]

        # Validate required fields
        if not all(k in results for k in ["name", "summary", "bad_group_flag"]):
            raise ValueError("Missing required fields in API response")

        return results

    except Exception as e:
        print(f"Error in label_single_group: {e}")
        return {
            "name": "Error Processing Group",
            "summary": f"Error occurred during processing: {str(e)}",
            "bad_group_flag": "1",
        }
