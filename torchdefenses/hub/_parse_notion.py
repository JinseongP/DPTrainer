import json
import requests

NOTION_API_TOKEN = 'secret_Adp7AI8ugJpicvcKQDAbjpuN7Ah6fbgJ6JnVilOHzFj'
NOTION_DATABASE_ID = "24c1f2e9ebc247daad8ce803289676d4"
NOTION_DATABASE_URL = f'https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query'


def get_text_from_res(res, key):
    return res[key]['title'][0]['plain_text']


def get_multi_from_res(res, key):
    items = []
    for item in res[key]['multi_select']:
        items.append(item['name'])
    return items


def get_value_from_res(res, key):
    return res[key]['number']


def get_checkbox_from_res(res, key):
    return res[key]['checkbox']


def get_link_from_res(res, key):
    urls = {}
    for item in res[key]['files']:
        name = item['name']
        url = item['file']['url']
        urls[name] = url
    return urls


def get_link_by_id(id, flag=""):
    headers = {'Authorization': f'Bearer {NOTION_API_TOKEN}',
               'Notion-Version': '2022-02-22',
               'Content-Type': 'application/json'}
    payload = {'page_size': 100}
    has_more = True
    while has_more is True:
        res = requests.post(NOTION_DATABASE_URL, data=json.dumps(payload), headers=headers).json()
        has_more = res['has_more']
        payload = {'start_cursor':res['next_cursor']}

        for r in res['results']:
            prop = r['properties']
            if len(prop["ID"]['title']) == 0:
                continue
            if get_text_from_res(prop, "ID") != id:
                continue
            print("="*50)
            print(id)
            print("="*50)
            print("Tags: "+", ".join(get_multi_from_res(prop, "Tags")))
            print("Dataset: "+", ".join(get_multi_from_res(prop, "Dataset")))
            print("Architecture: "+", ".join(get_multi_from_res(prop, "Architecture")))
            print("Extra Data:", get_checkbox_from_res(prop, "Extra Data"))
            for measure in ["Clean", "FGSM", "PGD", "AutoAttack"]:
                value = get_value_from_res(prop, measure+flag)
                if value:
                    print(measure+flag+": %2.2f%%"%value)
            print("Detail Info: "+r['url'])
            print("="*50)
            return get_link_from_res(prop, "Pretrained")
