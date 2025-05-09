
import great_expectations as gx
from src.config import ROOT_DIR

# We import the existing DataContext
context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

checkpoint = context.checkpoints.get("imdb_reviews_checkpoint")

checkpoint.run()