import os
import uuid
import asyncio
import concurrent.futures
import multiprocessing
import traceback
from typing import Dict, Optional, List, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
KEY = os.getenv("API")
genai.configure(api_key=KEY)

def remove_asterick(text):
    text = text.replace('*', '')
    return text

class OptimizedImageAnalysisManager:
    def __init__(self, max_workers=None):
        """
        Initialize the analysis manager with configurable worker pool
        
        Args:
            max_workers (int, optional): Number of workers for parallel processing
        """
        # Use CPU count minus 1, but at least 1 worker
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        self.max_workers = max_workers
        self.analysis_cache = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
    
    def _validate_image(self, image_path: str) -> bool:
        """
        Validate the uploaded image
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            bool: Whether the image is valid
        """
        try:
            with Image.open(image_path) as img:
                # Check image dimensions and size
                if img.width < 10 or img.height < 10:
                    raise ValueError("Image is too small")
                
                # Check file size (e.g., limit to 10MB)
                if os.path.getsize(image_path) > 10 * 1024 * 1024:
                    raise ValueError("Image size exceeds 10MB limit")
                
                return True
        except Exception as e:
            raise ValueError(f"Invalid image: {str(e)}")

    def _analyze_image_sync(self, image_path: str) -> str:
        """
        Synchronous image analysis method
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Analysis result
        """
        try:
            # Validate image first
            self._validate_image(image_path)
            
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            # Open image
            img = Image.open(image_path)
            
            # Comprehensive prompt
            prompt = """You are a seasoned and well experienced Crop Doctor, you are the best in the world and employed in my farm for crop Analysis. Here is an image of a crop, analyse this image, make diagnosis, what is here and  if it is a disease, the cause of the disease and step by step how to cure it
            Provide a comprehensive, structured analysis of this crop image with:
            sections = [
            "crop Name/Infection Name",
            "Precise Crop Identification",
            "Health Assessment", 
            "Root Cause", 
            "Treatment Recommendations", 
            "Preventive Strategies"
        ]"""
            
            # Generate content
            response = model.generate_content([prompt, img])
            return remove_asterick(response.text)
        
        except Exception as e:
            # Log the full traceback for internal debugging
            logger.error(f"Analysis error: {traceback.format_exc()}")
            raise RuntimeError(f"Crop analysis failed: {str(e)}")

    async def process_image(self, image_file: UploadFile) -> str:
        """
        Asynchronously process uploaded image
        
        Args:
            image_file (UploadFile): Uploaded image file
        
        Returns:
            str: Unique task ID for tracking analysis
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(image_file.filename)[1]
            ) as temp_file:
                temp_file.write(await image_file.read())
                temp_file_path = temp_file.name
            
            # Use thread pool for CPU-intensive task
            loop = asyncio.get_event_loop()
            analysis_result = await loop.run_in_executor(
                self.executor, 
                self._analyze_image_sync, 
                temp_file_path
            )
            
            # Store results in cache
            self.analysis_cache[task_id] = {
                "status": "completed",
                "filename": image_file.filename,
                "result": self._structure_response(analysis_result)
            }
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return task_id
        
        except ValueError as ve:
            # Specific validation errors
            self.analysis_cache[task_id] = {
                "status": "failed",
                "filename": image_file.filename,
                "error": str(ve)
            }
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image upload: {str(ve)}"
            )
        
        except Exception as e:
            # Unexpected errors
            self.analysis_cache[task_id] = {
                "status": "failed",
                "filename": image_file.filename,
                "error": "Unexpected processing error"
            }
            logger.error(f"Unexpected error: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, 
                detail="Image processing encountered an unexpected error"
            )

    def _structure_response(self, analysis_text: str) -> Dict[str, str]:
        """
        Structure the analysis text into a dictionary
        
        Args:
            analysis_text (str): Raw analysis text
        
        Returns:
            Dict[str, str]: Structured analysis response
        """
        # Define sections to extract
        sections = [
            "crop Name/Infection Name",
            "Precise Crop Identification",
            "Health Assessment", 
            "Root Cause", 
            "Treatment Recommendations", 
            "Preventive Strategies"
        ]
        
        # Initialize response dictionary
        structured_response = {}
        
        # Extract sections from text
        for section in sections:
            # Find section in text (case-insensitive)
            start_index = analysis_text.lower().find(section.lower())
            if start_index != -1:
                # Find next section or end of text
                next_section_indices = [
                    analysis_text.lower().find(s.lower(), start_index + len(section)) 
                    for s in sections if s != section
                ]
                next_section_indices = [idx for idx in next_section_indices if idx != -1]
                
                # Determine end of current section
                end_index = min(next_section_indices) if next_section_indices else len(analysis_text)
                
                # Extract section content
                content = analysis_text[start_index + len(section):end_index].strip(':- \n')
                
                # Add to response, converting to snake_case key
                key = section.lower().replace(' ', '_')
                structured_response[key] = content.strip()
        
        return structured_response

    def get_all_analyses(self) -> List[Dict]:
        """
        Retrieve all stored analyses with enhanced error handling
        
        Returns:
            List[Dict]: List of all analyses with their details
        """
        try:
            return [
                {
                    "task_id": task_id,
                    "filename": task_info.get("filename", "Unknown"),
                    "status": task_info["status"],
                    "result": task_info.get("result") if task_info["status"] == "completed" else None,
                    "error": task_info.get("error") if task_info["status"] == "failed" else None
                }
                for task_id, task_info in self.analysis_cache.items()
            ]
        except Exception as e:
            logger.error(f"Error retrieving analyses: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, 
                detail="Failed to retrieve analyses. Please try again."
            )

# Initialize FastAPI application with error handling
app = FastAPI(
    title="Advanced Crop Analysis API",
    description="AI-powered parallel image processing for agricultural insights",
    version="2.0.0"
)

# Add comprehensive CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize optimized image analysis manager
analysis_manager = OptimizedImageAnalysisManager()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for analysis with enhanced validation
    
    Args:
        file (UploadFile): Image file to analyze
    
    Returns:
        Dict with task ID for tracking analysis
    """
    try:
        # Strict file type validation
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. "
                       "Supported types: JPEG, PNG, WebP"
            )
        
        # Check file size before processing
        file_size = file.file.seek(0, 2)  # Move to end of file
        file.file.seek(0)  # Reset file pointer
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400, 
                detail="File size exceeds 10MB limit"
            )
        
        # Process image and get task ID
        task_id = await analysis_manager.process_image(file)
        return {"task_id": task_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected upload error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error during image upload"
        )

@app.get("/analysis/{task_id}")
async def get_analysis(task_id: str):
    """
    Retrieve analysis results for a given task ID with detailed error handling
    
    Args:
        task_id (str): Unique identifier for the analysis task
    
    Returns:
        Dict with analysis results or current status
    """
    try:
        # Validate task ID format
        try:
            uuid.UUID(task_id)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid task ID format"
            )
        
        # Check if task exists
        if task_id not in analysis_manager.analysis_cache:
            raise HTTPException(
                status_code=404, 
                detail="Analysis not found. Please upload an image first."
            )
        
        # Retrieve task result
        task_result = analysis_manager.analysis_cache[task_id]
        
        # Handle different task statuses
        if task_result['status'] == 'failed':
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {task_result.get('error', 'Unknown error')}"
            )
        
        if task_result['status'] == 'processing':
            return {"status": "processing", "message": "Analysis in progress"}
        
        return task_result['result']
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected analysis retrieval error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error retrieving analysis"
        )

@app.get("/analyses")
async def get_all_analyses():
    """
    Retrieve all stored analyses with comprehensive error handling
    
    Returns:
        List of all analyses with their details
    """
    try:
        # Get all analyses
        analyses = analysis_manager.get_all_analyses()
        
        # Check if there are any analyses
        if not analyses:
            raise HTTPException(
                status_code=404, 
                detail="No analyses found. Upload an image first."
            )
        
        return analyses
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected analyses retrieval error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="Unexpected error retrieving analyses"
        )

# Main execution for local development
# if __name__ == "__main__":
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0", 
#         port=8000, 
        
#     )