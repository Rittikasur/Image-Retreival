from fastapi import FastAPI, File, UploadFile
from mobilenetv1 import Mobilenetv1
import shutil
import os
import zipfile
app = FastAPI()
model = Mobilenetv1()
training_status = None
async def saveFile(fileobj):
    if(os.path.exists("./tmp/") != True):
        os.makedirs("./tmp")
    with open("./tmp/"+fileobj.filename, "wb") as buffer:
        shutil.copyfileobj(fileobj.file, buffer)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    await saveFile(file)
    output = model.search_image("./tmp/"+file.filename)
    return {"output": output}


@app.post("/uploaddataset/")
async def create_dataset(file: UploadFile):
    try: 
        if(os.path.exists("./data/") != True):
            os.makedirs("./data")
        with open("./data/"+file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        with zipfile.ZipFile("./data/"+file.filename, 'r') as zip_ref:
            zip_ref.extractall("./data")
        return {"dataset is Successfully Created"}
    except:
        shutil.rmtree('./data')
        return { "Some Error Occured"}


@app.post("/train/")
async def create_embedding():
    global training_status
    try:
        training_status = "Training"
        images_list = os.listdir("./data/all_data")[:100]
        model.populate_db("./data/all_data",images_list)
        training_status = "Training Completed"
        return {"Embeddings created Successfully"}
    except:
        training_status = "Error Occured while Training"
        return { "Some Error occured while creating embedding"}

@app.post("/trainingstatus/")
async def training_stat():
    global training_status
    return {"Training Status":training_status}
