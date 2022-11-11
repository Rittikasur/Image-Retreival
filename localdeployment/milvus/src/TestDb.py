from pymilvus import utility
from pymilvus import Collection
from pymilvus import connections,CollectionSchema, FieldSchema, DataType
import unittest
import Db
class TestDb(unittest.TestCase):
    def setUp(self):
        self.dbmodel = Db.DBClass()
        self.dbmodel.open_connection()
        
    def test_loadOrCreateconnection(self):
        #self.dbmodel.loadOrCreate_collection("ORGIndia")
        collection = self.dbmodel.getCollection("ORGIndia")
        self.assertEqual(type(collection),Collection)

    def test_EmbeddingInsert(self):
        self.dbmodel.loadOrCreate_collection("ORGIndia")
        mr = self.dbmodel.insertData(["HelloW"],[[ float(2) for _ in range(10)]])
        self.assertEqual(mr.insert_count,1)

    def test_EmbeddingSearch(self):
        self.dbmodel.loadOrCreate_collection("ORGIndia")
        rs = self.dbmodel.searchEmbedding([float(1) for _ in range(100)])
        print(rs)

if __name__=="__main__":
    unittest.main()
