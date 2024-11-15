import pickle
import os


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
