from sentence_transformers import SentenceTransformer
import time
from .utils import save_or_load_pickle


# Function to embed all tweets and time it
def embed_tweets(tweets_df, model):
    start_time = time.time()
    tweet_texts = tweets_df["emb_text"].tolist()
    embeddings = model.encode(tweet_texts, show_progress_bar=True)
    end_time = time.time()
    print(f"Embedding all tweets took {end_time - start_time:.2f} seconds")
    return embeddings


def load_or_make_embeddings(filepath, tweets_df):
    embeddings_path = filepath

    def create_embeddings():

        model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5", trust_remote_code=True
        ).cuda()
        return embed_tweets(tweets_df, model)

    embeddings = save_or_load_pickle(embeddings_path, create_embeddings)
    return embeddings
