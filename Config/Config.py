class Config:
    def __init__(self,model_name):
        if(model_name == "inception_v3"):
            print(model_name,"is being used")
            self.MILVUS_DBNAME="ORGIndia_inceptionv3"
            self.MODEL_WEIGHTS = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
            self.FEATURE_VECTOR_SIZE = 2048
            self.INPUT_DIMS = (299,299)
        elif(model_name == "inception_resnet_v2"):
            print(model_name,"is being used")
            self.MODEL_WEIGHTS = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5"
            self.INPUT_DIMS = (299,299)
            self.MILVUS_DBNAME="ORGIndia_inception_resnet_v2"
            self.FEATURE_VECTOR_SIZE = 1536
        else:
            print(model_name,"is being used")
            self.MILVUS_DBNAME="ORGIndia_inceptionv3"
            self.MODEL_WEIGHTS = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
            self.FEATURE_VECTOR_SIZE = 2048
            self.INPUT_DIMS = (299,299)