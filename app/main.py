from fastapi import FastAPI, UploadFile, HTTPException
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import io

app = FastAPI()

# Load model and labels at startup
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
categories = weights.meta["categories"]
model.eval()

preprocess = weights.transforms()


@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400, detail="Only JPEG, PNG, or WEBP images accepted"
        )

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image — file may be corrupt or empty",
        )

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)

    class_idx = probs.argmax(dim=1).item()

    label = categories[class_idx]
    confidence = probs[0][class_idx].item()

    return {"label": label, "confidence": f"{confidence:.2%}", "class_index": class_idx}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
