# Ep-det Machine Learning Webserver
## _Get predictions about your data_



This project is a flask webserver application that is used as an interface to a machine learning model. It allows us to get predictions for samples using HTTP methods.

## Prerequisites

- Docker Installed on your machine

## Running the Webserver

This app requires [Docker](https://www.docker.com/)  to run.

Build the docker file image

```sh
docker build -t ep-det-ml-model .
```

Run your Container 

```sh
docker run -p 2000:2000 ep-det-ml-model
```

You can verify that your server is running by visiting the following address on your browser
```
localhost:2000
```

## Sample Run

You can run a prediction for a sample.

by executing a post request to /prediction route
with the body in sample.json file 

Sample request example
```
POST localhost:2000/prediction
{
    "data": [
        -8,
        -10,
        19,
        20,
        ...
    ]
}
```

Sample response example
```
{
    "prediction": {
        "confidence": "0.99888843",
        "label": "C"
    }
}
```