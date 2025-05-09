# Great Expectations demo<!-- omit in toc -->
In this demo we will see the main features of [Great Expectations](https://greatexpectations.io/) to validate data.

## Contents <!-- omit in toc -->
- [Install Great Expectations](#install-great-expectations)
- [Configure your Great Expectations context](#configure-your-great-expectations-context)
  - [Connect the context to the data](#connect-the-context-to-the-data)
  - [Create an expectation suite](#create-an-expectation-suite)
  - [Creating a validaton](#creating-a-validaton)
  - [Creating a checkpoint](#creating-a-checkpoint)
- [Validate the data againts your expectations](#validate-the-data-againts-your-expectations)

## Install Great Expectations
The first step is to install Great Expectations. To do this, we can run the following command:
```bash
uv add great_expectations
```

## Configure your Great Expectations context
The next step is to configure our Great Expectations context. To do this, we first need to create a File Data Context. This will allow to persist the data context and use it in future sessions. For other types of data contexts, see the [Great Expectations documentation](https://docs.greatexpectations.io/docs/core/set_up_a_gx_environment/create_a_data_context).

To create a File Data Context, we run the following code:
```python
import great_expectations as gx

from src.config import INTERIM_DATA_DIR, ROOT_DIR

context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)
```

When you specify `mode='file'`, the `get_context(mode='file')` method instantiates and returns the first File Data Context it finds in the folder hierarchy of your current working directory.

If a File Data Context configuration is not found, `get_context(mode='file')` creates a new File Data Context in your current working directory and then instantiates and returns the newly created File Data Context.

### Connect the context to the data
Once we have our context, we can connect it to our data. In this case, we will connect it to the cleaned data stored as a Parquet file. For this, we need to create a Pandas datasource and add a parquet asset to it.

To create a Pandas datasource and add a dataframe asset to it, we can run the following code:
```python
datasource = context.data_sources.add_or_update_pandas(name="imdb_reviews")

data_asset = datasource.add_parquet_asset(name="imdb_reviews_parquet", path=INTERIM_DATA_DIR / "imdb_cleaned.parquet")
batch_definition = data_asset.add_batch_definition(name="imdb_reviews_data")
```

### Create an expectation suite
We can group our expectations on the data in expectation suites. An expectation suite is a collection of expectations that can be saved and reused across multiple batches of data.

To create an expectation suite and add it to our context, we can run the following code:
```python
expectation_suite = gx.ExpectationSuite("imdb_reviews_data_validation")
context.suites.add_or_update(expectation_suite)
```

Now we can simply add expectations to the suite. For a complete list of expectations, see the [Great Expectations page](https://greatexpectations.io/expectations/).

```python
# Explore available expectations at https://greatexpectations.io/expectations/
# Validate the "text" column
expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="text"))
expectation_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="text", type_="str"))
```

Finally, we can save the expectation suite to the context. This will allow us to reuse it in the future.
```python
expectation_suite.save()
```

### Creating a validaton
A validation is a fixed reference that joins a batch of data to an expectation suite.

To create a validation, we can run the following code:
```python
validation_definition = gx.ValidationDefinition(
    name="imdb_reviews_data_validator",
    data=batch_definition,
    suite=expectation_suite,
)
context.validation_definitions.add_or_update(validation_definition)
```

### Creating a checkpoint
A Checkpoint executes one or more Validation Definitions and then performs a set of Actions based on the Validation Results each Validation Definition returns.

For instance, we could tell the checkpoint to update the Data Docs after each validation.
```python
action_list = [
    gx.checkpoint.UpdateDataDocsAction(name="update_data_docs"),
]
validation_definitions = [validation_definition]

checkpoint = gx.Checkpoint(
    name="imdb_reviews_checkpoint",
    validation_definitions=validation_definitions,
    actions=action_list,
    result_format="SUMMARY",
)

context.checkpoints.add_or_update(checkpoint)
```

For a complete example of a possible configuration, see the [gx_context_configuration.py](../src/data/gx_context_configuration.py) file.

> See the [Great Expectations documentation](https://docs.greatexpectations.io/docs/core/set_up_a_gx_environment/create_a_data_context) for more information on how to configure your context.


## Validate the data againts your expectations
Now that we have our context configured, we can validate the data against our expectations. To do this, we simply need to
get the data context and get the checkpoint we created earlier. We can then run the checkpoint to validate the data.

To do this, we can run the following code:
```python
context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)
checkpoint = context.checkpoints.get("imdb_reviews_checkpoint")
checkpoint.run()
```

You can see an example of this in the [validate_data.py](../src/data/validate_data.py) file.

After running your validations, you can see the results in the Data Docs. By default, the Data Docs are stored in the `gx/uncommitted/data_docs` folder. You can change this by setting the `base_dir` parameter in the `data_docs_sites` section in the `great_expectations.yml` file.

You can see an example of the Data Docs in the [gx/data_docs](../gx/data_docs/) folder.