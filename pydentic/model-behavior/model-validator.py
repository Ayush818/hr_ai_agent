from pydantic import BaseModel, computed_field, field_validator, model_validator


class User(BaseModel):
    