import requests
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import json
from sklearn.metrics import f1_score
import json

def preprocess(df):
    texts = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            label = row["labels"]
            data = json.loads(row["data"])
            texts.append(data)
            labels.append(label)
        except Exception as e:
            pass
    return texts, labels


def test_slu_model(test_csv, service_url):
    df_test = pd.read_csv(test_csv)
    texts, true_labels = preprocess(df_test)
    predicted_labels = []
    for i in texts:
        req_obj = {}
        req_obj["alternatives"] = i["alternatives"]
        req_obj["context"] = {}
        req_obj["context"]["current_state"] = "test_state"
        API_ENDPOINT = service_url
        data = json.loads(json.dumps(req_obj))
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        r = requests.post(url=API_ENDPOINT, json=data, headers=headers)
        response = r.json()
        predicted_labels.append(response["response"]["intents"][0]["name"])

    print(classification_report(true_labels, predicted_labels))
    if f1_score(true_labels, predicted_labels, average="weighted") > 0.85:
        print("Pass")
    else:
        print(":/ Lets Try Again")


def test_duckling_entities(test_json, service_url):
    predicted_entities = []
    true_entities = []
    f = open(test_json)
    tests = json.load(f)
    for i in tests:
        req_obj = {}
        req_obj["alternatives"] = i["alternatives"]
        req_obj["context"] = {}
        req_obj["context"]["current_state"] = "test_state"
        API_ENDPOINT = service_url
        data = json.loads(json.dumps(req_obj))
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        r = requests.post(url=API_ENDPOINT, json=data, headers=headers)
        response = r.json()
        true_entities.append(i["entity_type"])
        try:
            predicted_entities.append(response["response"]["entities"][0]["entity_type"])
        except:
            predicted_entities.append("none")
    
    pass_flag = True 
    for i in range(len(true_entities)):
        if true_entities[i] != predicted_entities[i]:
            pass_flag = False
            print("failed test case:")
            print("true entity_type:", true_entities[i])
            print("predicted entity_type:", predicted_entities[i])
            print("description: ", tests[i]["description"])
        
    if pass_flag:
        print("All tests passed!")


