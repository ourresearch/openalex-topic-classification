import requests
import json

with open('api_key.json', 'r') as f:
    api_key = json.load(f)[0]['api-key']

with open('api_key.json', 'r') as f:
    request_url = json.load(f)[0]['url']

headers = {"X-API-Key": api_key}

def get_tags(context, iter_num):
    if isinstance(context, list):
        context = json.dumps(context)
    res = requests.post(request_url, json=context, headers=headers)
    output_api = res.text
    return iter_num, output_api
