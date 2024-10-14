import requests

'''
python 调用 0.zone api（信息系统）示例
'''

data = {
    "query": "国网冀北电力有限公司",
    "query_type": "site",
    "page": 10,
    "pagesize": 100,
    "zone_key_id": "6099d4d566ff74aa749a95b3697b6eed"
}

res = requests.post('https://0.zone/api/data/', json=data)

print(res.json())
w