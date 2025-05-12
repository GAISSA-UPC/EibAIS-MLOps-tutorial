import great_expectations as gx
from great_expectations import ValidationDefinition
from pytest import fixture

from src.config import ROOT_DIR
from src.data.gx_context_configuration import CLEAN_DATA_VALIDATOR


@fixture
def clean_data_validator() -> ValidationDefinition:
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)
    return context.validation_definitions.get(CLEAN_DATA_VALIDATOR)


def test_clean_data(clean_data_validator: ValidationDefinition):
    validation_result = clean_data_validator.run(result_format="BOOLEAN_ONLY")

    expectations_failed = validation_result["statistics"]["unsuccessful_expectations"]

    assert expectations_failed == 0, f"There were {expectations_failed} failing expectations."
