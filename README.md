# Image Classification API with FastAPI

This project provides an image classification API built with FastAPI and PyTorch using a pretrained ResNet50 model trained on the ImageNet dataset.

The API accepts image uploads, performs deep learning inference, and returns:
- predicted image label
- confidence score
- ImageNet class index

The application uses pretrained ImageNet weights from torchvision, so no custom model training is required.

---

## Features

- Image upload prediction endpoint
- Pretrained ResNet50 inference
- FastAPI REST API
- Docker support
- GitHub Actions CI workflow
- Interactive Swagger UI documentation

---

## Project Structure

```txt
project/
│
├── app/
│   └── main.py
│
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

---

## Installation Guide

Clone the repository

```bash
$ git clone git@github.com:HonourJAH/image-classification-api.git
```

Navigate into the project directory

```bash
$ cd image-classification-api
```

Create and activate a virtual environment

### Linux/macOS

```bash
$ python3 -m venv venv

$ source venv/bin/activate
```

### Windows

```bash
$ python -m venv venv

$ venv\Scripts\activate
```

Verify the virtual environment is active

```bash
$ which python

/path/to/your/project/venv/bin/python
```

Upgrade pip

```bash
$ python -m pip install --upgrade pip
```

Install dependencies

```bash
$ pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI development server

```bash
$ fastapi dev app/main.py
```

Or using uvicorn directly

```bash
$ uvicorn app.main:app --reload
```

The API will be available at:

[api-end-point](http://127.0.0.1:8000)

---

## API Documentation

FastAPI automatically generates interactive API documentation.

Visit:

- [Swagger UI](http://127.0.0.1:8000/docs)

- [ReDoc](http://127.0.0.1:8000/redoc)

---

## Example Prediction Request

Send an image to the prediction endpoint

```bash
$ curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test.jpg"
```

---

## Example API Response

```json
{
  "label": "tabby cat",
  "confidence": "97.42%",
  "class_index": 281
}
```

---

## Health Check Endpoint

Verify the API is running correctly

```bash
$ curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy"
}
```

---

## Running with Docker

Build the Docker image

```bash
$ docker build -t image-classification-api .
```

Run the container

```bash
$ docker run -d -p 8000:8000 image-classification-api
```

Verify the API is running

```bash
$ curl http://127.0.0.1:8000/health
```

---

## Supported Image Formats

The API currently accepts:

- JPEG
- PNG
- WEBP

---

## Technologies Used

- Python
- FastAPI
- PyTorch
- Torchvision
- Pillow
- Docker
- GitHub Actions

---

## CI/CD

This project uses GitHub Actions to:

- build the Docker image
- start the API container
- run health check tests automatically on every push and pull request

---

## Future Improvements

- Top-k predictions
- GPU inference support
- Batch image prediction
- Authentication & rate limiting
- Kubernetes deployment
- Model monitoring
