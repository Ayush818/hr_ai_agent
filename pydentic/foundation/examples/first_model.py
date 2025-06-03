from pydantic import BaseModel


class User(BaseModel):
    id: int
    name: str
    is_active: bool


class Product(BaseModel):
    id: int
    name: str
    price: float
    is_stock: bool


input_product = {"id": 1, "name": "Laptop", "price": 999.99, "is_stock": True}
product = Product(**input_product)
print(product)


input_data = {"id": 1, "name": "Aayush Raj Pokhrel", "is_active": True}
user = User(**input_data)

print(user)

# user = User(**input_data)
# print(user)
