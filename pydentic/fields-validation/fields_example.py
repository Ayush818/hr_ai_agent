from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Cart(BaseModel):
    user_id: int
    items: List[str]
    quantities: Dict[str, int]


class BlogPost(BaseModel):
    title: str
    content: str
    image_url: Optional[str] = None


# by using typing only
class Employee(BaseModel):
    id: int
    name: str
    department: Optional[str] = "General"
    salary: float


# using typing and pydantic[pydantic.fields]]


class EmployeeWithField(BaseModel):
    id: int
    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Name of the employee",
        examples="John Doe",
    )
    department: Optional[str] = Field(
        "General",
        description="Department of the employee",
        examples=["HR", "Engineering", "Sales"],
    )
    salary: float = Field(
        ...,
        gt=0,
        description="Salary of the employee",
        examples=[50000.0, 75000.0, 100000.0],
    )
