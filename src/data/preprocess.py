import polars as pl

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def sanitize_text(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitize the text text by removing special characters and normalizing.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input DataFrame containing the text text.

    Returns
    -------
    pl.DataFrame
        The sanitized DataFrame with cleaned text text.
    """
    dataframe = dataframe.with_columns(
        [
            pl.col("text").str.replace(r"(\n|\r|\s+)", " ").str.strip_chars().str.normalize("NFKD"),
        ]
    )
    return dataframe


def remove_empty_or_duplicate(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Remove empty or duplicate rows from the DataFrame.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The input DataFrame containing the text text and labels.

    Returns
    -------
    pl.DataFrame
        The cleaned DataFrame with no duplicate or empty rows.
    """
    dataframe = dataframe.filter(pl.col("text").is_not_null() & pl.col("labels").is_not_null())
    dataframe = dataframe.filter(pl.col("text") != "")
    return dataframe.unique(subset=["text", "labels"])


def preprocess_data():
    """
    Preprocess the raw data and save it to the 'data/interim' directory.
    """
    df = pl.read_parquet(RAW_DATA_DIR / "imdb.parquet")
    df = df.rename({"label": "labels"})

    df = sanitize_text(df)

    df = remove_empty_or_duplicate(df)

    df.write_parquet(INTERIM_DATA_DIR / "imdb_cleaned.parquet")


if __name__ == "__main__":
    preprocess_data()
