from pydantic import BaseModel, Field
from typing import List


class ImageItem(BaseModel):
    
    """
    A Pydantic model representing an image in base64 format.

    Attributes:
    ----------
    Imageb64 : str
        A string representing the image in base64 format.
    """
    
    Imageb64: str

class DataFrameResponse(BaseModel):
    
    """
    A Pydantic BaseModel representing the response of an object detection API that processes an image and returns 
    a list of object detections.

    Attributes:
    -----------
    Imageb64 : str
        A string representing the base64-encoded image.
    Response : List[DataFrameResponse]
        A list of DataFrameResponse objects representing the detected objects in the image.
    """
    
    name: str = Field(..., max_length=30)
    class_id: int = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1.0, description="confidence in percentage.")
    xmin: float = Field(..., ge=0, le=5000, description="xmin in pixel.")
    ymin: float = Field(..., ge=0, le=5000, description="ymin in pixel.")
    xmax: float = Field(..., ge=0, le=5000, description="xmax in pixel.")
    ymax: float = Field(..., ge=0, le=5000, description="ymax in pixel.")
    
class SPADResponse(BaseModel):
    Imageb64: str
    Response: List[DataFrameResponse]
    