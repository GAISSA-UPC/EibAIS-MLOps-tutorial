import polars as pl

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SEED, TRAIN_SPLIT, VALIDATION_SPLIT


def split_data():
    """
    Split the cleaned data into train, validation, and test sets and save them to the 'data/processed' directory.

    The split is done in the ratio of 70% train, 20% validation, and 10% test.
    The random seed is set to ensure reproducibility.
    """
    df = pl.read_parquet(INTERIM_DATA_DIR / "imdb_cleaned.parquet")

    n_samples = df.height
    train_size = int(n_samples * TRAIN_SPLIT)
    validation_size = int(n_samples * VALIDATION_SPLIT)

    train_df = df.sample(n=train_size, with_replacement=False, seed=SEED)
    remaining_df = df.join(train_df, on="review", how="anti")
    validation_df = remaining_df.sample(n=validation_size, with_replacement=False, seed=SEED)
    test_df = remaining_df.join(validation_df, on="review", how="anti")

    train_df.write_parquet(PROCESSED_DATA_DIR / "train.parquet")
    validation_df.write_parquet(PROCESSED_DATA_DIR / "validation.parquet")
    test_df.write_parquet(PROCESSED_DATA_DIR / "test.parquet")


if __name__ == "__main__":
    split_data()
