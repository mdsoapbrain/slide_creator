from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime


def test_connection(client):
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
def get_collection(client, collection_name, recreate=False):
    mydb = client["SushiGo"]
    collist = mydb.list_collection_names()
    if collection_name in collist and not recreate:
        print(f"The collection - {collection_name} - exists.")
        return mydb[collection_name]
    else:
        print(f"The collection - {collection_name} - has been created.")
        if collection_name in collist and recreate:
            mydb[collection_name].drop()
        return mydb[collection_name]

def add_doc(mycol, username, password):
    res = mycol.insert_one({username: password})
    print("Added user with id: ", res.inserted_id)

def match_doc(mycol, username, password):
    res = mycol.find_one({username: password})
    return res

def view_all_docs(mycol, max_items=100):
    cursor = mycol.find({})
    for document in cursor[:max_items]:
        print(document) 

def get_budget_parameter(mycol, parameter):
    res = mycol.find_one({"parameter": parameter})
    return res['value']

def update_budget_parameter(mycol, parameter, new_val):
    mycol.update_one({"parameter": parameter}, 
                        {"$set": {"value": new_val}}
                        )
    
def update_user_QA(coll, member_id, new_filename, new_question, new_answer):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    coll.update_one({'member_id': member_id}, {'$push': {'qa_timestamp': timestamp}})
    coll.update_one({'member_id': member_id}, {'$push': {'filename': new_filename}})
    coll.update_one({'member_id': member_id}, {'$push': {'questions': new_question}})
    coll.update_one({'member_id': member_id}, {'$push': {'response': new_answer}})
