from pymilvus import CollectionSchema, FieldSchema, DataType

class ModelSchema:
  def __init__(self,dimensions):

    self.sku_id = FieldSchema(
      name="sku_id", 
      dtype=DataType.INT64, 
      is_primary=True, 
    )
    self.sku_name = FieldSchema(
      name="sku_name", 
      dtype=DataType.VARCHAR, 
      max_length=200,
    )
    self.sku_embedding = FieldSchema(
      name="sku_embedding", 
      dtype=DataType.FLOAT_VECTOR, 
      dim=dimensions
    )

  def CreateSchema(self):
    self.schema = CollectionSchema(
      fields=[self.sku_id, self.sku_name, self.sku_embedding], 
      auto_id	= True,
      description="ORG India Embedding Search"
    )
    return self.schema