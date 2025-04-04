from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import torch
import asyncio
from typing import Optional, Dict, Any, List
import time

# Import the functions from your existing phi4_tester.py
from phi4_tester import load_phi4_model, generate_response

# Create FastAPI app
app = FastAPI(
    title="Phi-4 API",
    description="REST API for local Phi-4 language model inference",
    version="1.0.0"
)

# Global variables to store model and tokenizer
model = None
tokenizer = None
model_loading = False
model_loaded = False

# Define request and response models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    generated_length: int
    processing_time: float

class ModelInfoResponse(BaseModel):
    model_loaded: bool
    device: str
    cuda_available: bool
    gpu_info: Optional[Dict[str, Any]] = None

class ModelLoadRequest(BaseModel):
    model_path: str
    use_gpu: bool = True

class ModelLoadResponse(BaseModel):
    success: bool
    message: str
    model_info: Optional[ModelInfoResponse] = None

async def load_model_async(model_path: str, use_gpu: bool):
    global model, tokenizer, model_loading, model_loaded
    model_loading = True
    
    try:
        # Load the model in a non-blocking way
        model, tokenizer = load_phi4_model(model_path=model_path, use_gpu=use_gpu)
        model_loaded = True
        model_loading = False
    except Exception as e:
        model_loading = False
        model_loaded = False
        raise e

@app.get("/", response_model=Dict[str, str])
async def root():
    """Get API information."""
    return {
        "name": "Phi-4 API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "/": "This information",
            "/load": "Load the Phi-4 model",
            "/generate": "Generate text using the model",
            "/model-info": "Get information about the loaded model"
        }
    }

@app.post("/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """Load the Phi-4 model with specified parameters."""
    global model_loading, model_loaded
    
    if model_loading:
        return ModelLoadResponse(
            success=False,
            message="Model is already being loaded. Please wait."
        )
    
    if model_loaded:
        return ModelLoadResponse(
            success=False,
            message="Model is already loaded. Unload it first if you want to load a different configuration."
        )
    
    try:
        # Start loading the model in the background
        background_tasks.add_task(load_model_async, request.model_path, request.use_gpu)
        
        return ModelLoadResponse(
            success=True,
            message=f"Model loading started in background from path: {request.model_path}. Check /model-info for status."
        )
    except Exception as e:
        return ModelLoadResponse(
            success=False,
            message=f"Error starting model loading: {str(e)}"
        )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the currently loaded model."""
    global model, model_loaded, model_loading
    
    info = {
        "model_loaded": model_loaded,
        "loading_in_progress": model_loading,
        "cuda_available": torch.cuda.is_available(),
        "device": "Loading..." if model_loading else "Not loaded"
    }
    
    if model_loaded and model is not None:
        device = next(model.parameters()).device
        info["device"] = str(device)
        
        if device.type == "cuda":
            info["gpu_info"] = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
            }
    
    return info

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the Phi-4 model."""
    global model, tokenizer, model_loaded
    
    if not model_loaded or model is None or tokenizer is None:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded. Call /load endpoint first."
        )
    
    try:
        start_time = time.time()
        
        # Use the existing generate_response function from phi4_tester.py
        output_text = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return GenerateResponse(
            text=output_text,
            generated_length=len(output_text),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {str(e)}"
        )

@app.delete("/unload")
async def unload_model():
    """Unload the model from memory."""
    global model, tokenizer, model_loaded, model_loading
    
    if model_loading:
        return {"success": False, "message": "Model is currently loading. Please wait before unloading."}
    
    if not model_loaded:
        return {"success": False, "message": "No model is currently loaded."}
    
    try:
        # Delete model and tokenizer to free up memory
        del model
        del tokenizer
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = None
        tokenizer = None
        model_loaded = False
        
        return {"success": True, "message": "Model unloaded successfully."}
    
    except Exception as e:
        return {"success": False, "message": f"Error unloading model: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("phi4_api:app", host="0.0.0.0", port=8000, reload=True)