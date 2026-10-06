import requests

url = "http://127.0.0.1:8000/api/v1/extraction/process"
files = {'file': open('inputs/input drawing _ naked shell.pdf', 'rb')}

response = requests.post(url, files=files)
import json
print(json.dumps(response.json(), indent=2))
