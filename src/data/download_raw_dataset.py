import polars as pl

from src.config import RAW_DATA_DIR


def download_dataset():
    """
    Download the dataset from Hugging Face and save it to the 'data/raw' directory.
    """
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
    }
    train_df = pl.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
    test_df = pl.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

    df = pl.concat([train_df, test_df], how="vertical")
    df.write_parquet(RAW_DATA_DIR / "imdb.parquet")


if __name__ == "__main__":
    download_dataset()
