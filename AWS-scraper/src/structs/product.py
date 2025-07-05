from typing import List, Optional
from pydantic import BaseModel

class Product(BaseModel):
    id: str
    name: str
    link: str
    price: float
    brand: Optional[str]
    images: List[str]
