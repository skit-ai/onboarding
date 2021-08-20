import requests
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report
import json
from sklearn.metrics import f1_score


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
