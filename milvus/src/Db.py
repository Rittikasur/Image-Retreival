import logging
import random
from pymilvus import Collection
from pymilvus import utility
from milvus.src.Schema import ModelSchema
# from Schema import ModelSchema
from pymilvus import connections

COLLECTION_NAME="ORGIndia"

class DBClass:
    def __init__(self):
        self.collection = None
        self.index_params = {
            "metric_type":"L2",
            "index_type":"IVF_FLAT",
            "params":{"nlist":1024}
            }


    def open_connection(self,host="localhost",port="19530"):
        connections.connect(
        alias="default", 
        host=host, 
        port=port
        )

    def getCollection(self,CollectionName):
        if(self.collection is None):
            self.loadOrCreate_collection(CollectionName)
        return self.collection

    def loadOrCreate_collection(self,Collection_name,dimension=100,indexed_column="sku_embedding"):
        if(utility.has_collection(Collection_name)):
            if(self.collection is None):
                self.collection = Collection(Collection_name)
        else:
            modelSchema = ModelSchema(dimensions=dimension)
            self.collection = Collection(
                name=Collection_name, 
                schema=modelSchema.CreateSchema(), 
                using='default', 
                shards_num=2,
                )
            self.collection.create_index(
                        field_name=indexed_column, 
                        index_params=self.index_params
                )

    def insertData(self,name,embedding):
        print("inserting Data")
        data = [name,embedding]
        mr = self.collection.insert(data)
        return(mr)

    def searchEmbedding(self,embedding,search_column="sku_embedding"):
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        self.collection.load()
        results = self.collection.search(
                        data=[embedding], 
                        anns_field=search_column, 
                        param=search_params, 
                        limit=5, 
                        expr=None,
                        consistency_level="Strong"
                    )
        self.collection.release()
        return results

    def querywithId(self,list_of_ids):
        self.collection.load()
        expression = "sku_id " +"in [" +",".join(map(str,list_of_ids)) +"]"
        res = self.collection.query(
            expr = expression, 
            output_fields = ["sku_name"],
            consistency_level="Strong"
            )
        self.collection.release()
        return res

