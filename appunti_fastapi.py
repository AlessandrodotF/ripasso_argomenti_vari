# 1) baic command to run fast api app:
# " fastapi dev file.py "

# * init the class FastApi

from fastapi import FastAPI, Depends
from enum import Enum


class Modelname(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


# @app.post()
# @app.delete()
@app.get("/")
async def root():
    return {"message": "ciao bro"}


@app.get("/item/{item_id}")  # è come get_chat_id
async def read_item(item_id: str):
    return {"item_id": item_id}


@app.get("/model_name/{model_name}")
async def get_model_name(model_name: Modelname):
    if model_name is Modelname.alexnet:  # funziona sia così
        return {"model_name": model_name, "message": "alexnet model bro"}

    if model_name.value == "resnet":  # che così
        return {"model_name": model_name, "message": "resnet model bro"}

    if model_name is Modelname.lenet:
        return {"model_name": model_name, "message": "lenet model bro"}

    else:
        return {"model_name": model_name, "message": "bho model bro"}


fake_items_db = [
    {"0": "0"},
    {"1": "1"},
    {"2": "2"},
    {"3": "3"},
    {"4": "4"},
    {"5": "5"},
    {"6": "6"},
    {"7": "7"},
    {"8": "8"},
    {"9": "9"},
    {"10": "10"},
    {"11": "11"},
]


def get_db():
    return fake_items_db  # è solo per il pattern


@app.get("/items/")
def read_items(min_id: int, max_id: int, db=Depends(get_db)):
    return db[min_id : max_id + min_id]
