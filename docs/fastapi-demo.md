# FastAPI demo<!-- omit in toc -->
In this demo we will see the main features of [FastAPI](https://fastapi.tiangolo.com/) to create an API for a simple
machine learning project.

The scrips used in this demo are based on the [SE4AI2021Course_FastAPI-demo](https://github.com/se4ai2122-cs-uniba/SE4AI2021Course_FastAPI-demo) GitHub project.

## Contents <!-- omit in toc -->
- [Install FastAPI](#install-fastapi)
- [Creating a simple API](#creating-a-simple-api)
- [Start the server](#start-the-server)
- [Try the API](#try-the-api)
  - [Access the API documentation](#access-the-api-documentation)
  - [Try some requests](#try-some-requests)
- [Test the API](#test-the-api)


## Install FastAPI
The first step is to install FastAPI and Uvicorn, which is a fast ASGI server (it can run asynchronous code in a single
process) to launch our application. To do this, we can run the following command:

```bash
uv add fastapi "uvicorn[standard]"
```

## Creating a simple API
We can now create a simple API. To do this, we create a new file called [`api.py`](../src/api/api.py) in the `src/api`
directory. In this file, we will create a FastAPI application and two endpoints:
- `/` will be the root endpoint, which will return a welcome message;
- `/predict` will return the prediction using the model specified in the path parameter.

Since the `/predict` endpoint receives a payload specifying the review, we will create a Pydantic
class called `PredictRequest` to represent our payload. We will use this class to validate that the input of the users is correct. This class will be located in the
[`schemas.py`](../src/api/schemas.py) file.


## Start the server
Use the following command to start the server:

```bash
uvicorn src.api.api:app \
    --host 0.0.0.0 \
    --port 5000 \
    --reload \
    --reload-dir src/api \
    --reload-dir models
```
In detail:

- `uvicorn src.api.api:app` is the location of app (`src` directory > `api` directory > `api.py` script > `app` object);
- `--reload` makes the server reload every time we update;
- `--reload-dir app` makes it only reload on updates to the `api/` directory;
- `--reload-dir models` makes it also reload on updates to the `models/` directory;

## Try the API
We can now test that the application is working. These are some of the possibilities:

- Visit [localhost:5000](http://localhost:5000/)
- Use `curl`

  ```bash
  curl -X GET http://localhost:5000/
  ```

- Access the API programmatically, e.g.:

  ```python
  import json
  import requests

  response = requests.get("http://localhost:5000/")
  print(json.loads(response.text))
  ```

- Use an external tool like [Postman](https://www.postman.com), which lets you execute and manage tests that can be
saved and shared with others.

### Access the API documentation
You can access the [Swagger UI](https://swagger.io/tools/swagger-ui/) in http://localhost:5000/docs for documentation
endpoint and select one of the models. The documentation generated via [Redoc](https://github.com/Redocly/redoc) is
accessible at the `/redoc` endpoint.


<center><figure>
  <img
  src="static/01_api_ui.png"
</figure></center>
<p style="text-align: center;">API User Interface in localhost:5000/docs endpoint.</p>

### Try some requests
To try an API request, click on the "Try it out" button and click execute.

For example:
#### Positive review <!-- omit in toc -->

```json
{
  "reviews": [
    {
      "review": "This is a great film. I loved it!"
    }
  ]
}
```

#### Positive and negative reivews <!-- omit in toc -->

```json
{
  "reviews": [
    {
      "review": "This is a great film. I loved it!"
    },
    {
      "review": "It was so hard not to fall asleep during this movie. It was too boring!"
    }
  ]
}
```

## Test the API
We can now test the API using [Pytest](https://docs.pytest.org/en/6.2.x/). To do this, we create a new file called [`test_api.py`](../src/tests/test_api.py) in the `tests/` directory.

Here we will create a fixture called `client` that will be used to test the API. We will also create a second fixture called `payload` that will be used to test the `/predict/tabular/{type}` endpoint. Since our endpoints expect the payload in JSON format we must build the `payload` return value according to the same format. If you have correctly implemented your API, the `/docs` endpoint will show you an example of the payload expected by each of your endpoints.

Finally, we will create a test for each endpoint. To do this, we will use the `client` fixture to make requests to the API and check the response.
