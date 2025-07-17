from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
from ml_service.models import MNISTModel

app = FastAPI(title="MNIST Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
model = MNISTModel()
# Load the best checkpoint
checkpoint = torch.load('logs/lightning_logs/version_0/checkpoints/mnist-epoch=03-val_loss=0.02.ckpt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    # Invert colors (MNIST has white digits on black background)
    image = 1.0 - image
    # Add batch dimension
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        input_tensor = preprocess_image(contents)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            probabilities = torch.softmax(output, dim=1)[0].tolist()
        
        return JSONResponse({
            "prediction": int(prediction),
            "probabilities": [float(p) for p in probabilities]
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "Welcome to MNIST Classification API. Use /predict endpoint with an image file to get predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 