import pickle
import os
import toolz as tz


from functools import wraps
from typing import Callable
import pandas as pd
import time


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


def save_or_load_pickle(file_path, data_func, *args, rerun=False, **kwargs):
    try:
        if rerun:
            print("Enforced running of clustering again")
            raise Exception("rerun")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Data loaded from {file_path}.")
    except Exception:
        print(f"Data not found at {file_path}. Running data function...")
        data = data_func(*args, **kwargs)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    return data


def pick(allowlist, d):
    return tz.keyfilter(lambda k: k in allowlist, d)


# Create error result object
def make_error_result(
    cluster_id,
    text_results,
    error_msg,
):
    return {
        "is_error": True,
        "error": str(error_msg),
        "message": text_results,
        "cluster_id": cluster_id,
        "cluster_summary": {
            "name": f"Error: {error_msg[:30]}...",
            "summary": str(error_msg),
        },
        "ontology_items": {},
        "low_quality_cluster": "1",
    }
