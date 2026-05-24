import torch
from fastapi.testclient import TestClient
from fastapi import status
from PIL import Image
import io
import numpy as np

from app.main import app, categories, model, preprocess

client = TestClient(app)


# ─── Helpers ────────────────────────────────────────────────────────────────


def make_image_bytes(format="JPEG", size=(224, 224), color=(255, 0, 0)):
    """Create a real in-memory image and return its raw bytes."""
    image = Image.fromarray(np.uint8(np.full((*size, 3), color)))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


def upload(image_bytes, filename="test.jpg", content_type="image/jpeg"):
    """Post an image to /predict and return the response."""
    return client.post(
        "/predict",
        files={"file": (filename, image_bytes, content_type)},
    )


# ─── Health Check ────────────────────────────────────────────────────────────


class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_returns_correct_body(self):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


# ─── Happy Path ──────────────────────────────────────────────────────────────


class TestPredictSuccess:
    def test_jpeg_returns_200(self):
        response = upload(make_image_bytes("JPEG"), "photo.jpg", "image/jpeg")
        assert response.status_code == status.HTTP_200_OK

    def test_png_returns_200(self):
        response = upload(make_image_bytes("PNG"), "photo.png", "image/png")
        assert response.status_code == status.HTTP_200_OK

    def test_webp_returns_200(self):
        response = upload(make_image_bytes("WEBP"), "photo.webp", "image/webp")
        assert response.status_code == status.HTTP_200_OK

    def test_response_has_label_field(self):
        response = upload(make_image_bytes())
        assert "label" in response.json()

    def test_response_has_confidence_field(self):
        response = upload(make_image_bytes())
        assert "confidence" in response.json()

    def test_response_has_class_index_field(self):
        response = upload(make_image_bytes())
        assert "class_index" in response.json()

    def test_label_is_string(self):
        response = upload(make_image_bytes())
        assert isinstance(response.json()["label"], str)

    def test_label_is_valid_imagenet_category(self):
        response = upload(make_image_bytes())
        assert response.json()["label"] in categories

    def test_confidence_is_percentage_string(self):
        response = upload(make_image_bytes())
        confidence = response.json()["confidence"]
        assert isinstance(confidence, str)
        assert confidence.endswith("%")

    def test_confidence_percentage_is_valid_number(self):
        response = upload(make_image_bytes())
        confidence = response.json()["confidence"]
        value = float(confidence.replace("%", ""))
        assert 0.0 <= value <= 100.0

    def test_class_index_is_integer(self):
        response = upload(make_image_bytes())
        assert isinstance(response.json()["class_index"], int)

    def test_class_index_is_within_imagenet_range(self):
        response = upload(make_image_bytes())
        class_index = response.json()["class_index"]
        assert 0 <= class_index <= 999

    def test_class_index_matches_label(self):
        response = upload(make_image_bytes())
        data = response.json()
        assert categories[data["class_index"]] == data["label"]

    def test_rgba_png_is_handled(self):
        """PNG with alpha channel should be converted to RGB without crashing."""
        image = Image.fromarray(np.uint8(np.full((224, 224, 4), 128)), mode="RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        response = upload(buffer.read(), "rgba.png", "image/png")
        assert response.status_code == status.HTTP_200_OK

    def test_small_image_is_handled(self):
        """Images smaller than 224x224 should be handled by the transforms."""
        response = upload(make_image_bytes(size=(32, 32)))
        assert response.status_code == status.HTTP_200_OK

    def test_large_image_is_handled(self):
        """Large images should be resized correctly by the transforms."""
        response = upload(make_image_bytes(size=(1024, 1024)))
        assert response.status_code == status.HTTP_200_OK

    def test_non_square_image_is_handled(self):
        response = upload(make_image_bytes(size=(300, 150)))
        assert response.status_code == status.HTTP_200_OK

    def test_same_image_returns_same_prediction(self):
        """Model is deterministic — same input should always produce same output."""
        image_bytes = make_image_bytes()
        response1 = upload(image_bytes)
        response2 = upload(image_bytes)
        assert response1.json() == response2.json()


# ─── Validation Errors ───────────────────────────────────────────────────────


class TestPredictValidation:
    def test_gif_returns_400(self):
        response = upload(make_image_bytes(), "anim.gif", "image/gif")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_pdf_returns_400(self):
        response = upload(b"%PDF-1.4 fake content", "doc.pdf", "application/pdf")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_plain_text_returns_400(self):
        response = upload(b"just some text", "note.txt", "text/plain")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_json_returns_400(self):
        response = upload(b'{"key": "value"}', "data.json", "application/json")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_error_detail_message_is_correct(self):
        response = upload(b"fake", "file.gif", "image/gif")
        assert response.json()["detail"] == "Only JPEG, PNG, or WEBP images accepted"

    def test_empty_file_raises_error(self):
        """An empty file should not crash the server."""
        response = upload(b"", "empty.jpg", "image/jpeg")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_corrupt_image_raises_error(self):
        """Random bytes labeled as JPEG should not crash the server."""
        response = upload(b"not_an_image_at_all", "corrupt.jpg", "image/jpeg")
        assert response.status_code == status.HTTP_400_BAD_REQUEST


# ─── Model & Categories Sanity Checks ────────────────────────────────────────


class TestModelSanity:
    def test_categories_has_1000_classes(self):
        assert len(categories) == 1000

    def test_categories_are_all_strings(self):
        assert all(isinstance(c, str) for c in categories)

    def test_model_is_in_eval_mode(self):
        assert not model.training

    def test_model_output_shape(self):
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 1000)

    def test_softmax_sums_to_one(self):
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
            probs = torch.softmax(output, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_preprocess_output_shape(self):
        image = Image.fromarray(np.uint8(np.zeros((224, 224, 3))))
        tensor = preprocess(image)
        assert tensor.shape == (3, 224, 224)

    def test_known_category_exists(self):
        assert "goldfish" in categories

    def test_first_category_is_tench(self):
        assert categories[0] == "tench"
