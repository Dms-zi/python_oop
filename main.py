import src.product

product_1 = src.product.Product("fore")


from src.product import Product

product_1 = Product("fore")

test_1 = """
>>> from src.products import Product
>>> product_1 = Product("fore")
>>> print(product_1.name)
"""