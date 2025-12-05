import sys
from pathlib import Path
import shutil
import os

# Add src to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from inference.infer_mfcc import GenreClassifier
except ImportError:
    from src.inference.infer_mfcc import GenreClassifier

app = FastAPI(title="Music Genre Classifier")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier
classifier = None

@app.on_event("startup")
async def startup_event():
    global classifier
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "outputs" / "mfcc_cnn" / "mfcc_cnn_trained.pth"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python src/training/train_mfcc.py")
        raise RuntimeError("Model file not found")
    
    try:
        classifier = GenreClassifier(str(model_path))
        print(f"Classifier initialized successfully")
        print(f"   Genres: {classifier.genres}")
        print(f"   Expected MFCC shape: ({classifier.expected_mfcc_vectors_per_segment}, 13)")
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        raise

@app.get("/health")
async def health():
    """Health check endpoint"""
    if classifier:
        return {"status": "healthy", "model": "loaded", "genres": classifier.genres}
    return {"status": "error", "model": "not loaded"}

@app.get("/genres")
async def get_genres():
    """Get list of supported genres"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    return {"genres": classifier.genres}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized. Server may still be starting up.")
    
    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save temporary file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Processing: {file.filename}")
        
        # Predict
        result = classifier.predict(str(temp_file_path), return_probabilities=True)
        
        print(f"Result: {result['genre']} ({result['confidence']*100:.1f}%)")
        
        # Clean up
        os.remove(temp_file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        if temp_file_path.exists():
            os.remove(temp_file_path)
        
        print(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# Mount static files for frontend
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
