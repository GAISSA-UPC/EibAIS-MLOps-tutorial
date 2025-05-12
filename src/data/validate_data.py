import great_expectations as gx
from loguru import logger

from src.config import ROOT_DIR
from src.data.gx_context_configuration import CHECKPOINT

# We import the existing DataContext
context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

checkpoint = context.checkpoints.get(CHECKPOINT)

checkpoint_result = checkpoint.run()

# Extract the Validation Result object from the Checkpoint results.
validation_result = checkpoint_result.run_results[list(checkpoint_result.run_results.keys())[0]]

expectations_run = validation_result["statistics"]["evaluated_expectations"]
expectations_failed = validation_result["statistics"]["unsuccessful_expectations"]

logger.info(f"Validation results: {expectations_run} expectations evaluated, {expectations_failed} failed.")
