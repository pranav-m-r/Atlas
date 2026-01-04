import requests

url = "http://192.168.1.50:8000/send"
data = {"text": "Hello Pi, this is an API call"}

r = requests.post(url, json=data)
print(r.json())
