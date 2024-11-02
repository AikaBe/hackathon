import requests
import json

url = 'http://routing.api.2gis.com/routing/7.0.0/global?key=90908d6d-9b1a-4234-b726-7f8c651e5374'
headers = {
    'Content-Type': 'application/json'
}

data = {
    "points": [
        {
            "type": "stop",
            "lon": 37.582591,
            "lat": 55.775364
        },
        {
            "type": "stop",
            "lon": 37.579206,
            "lat": 55.774362
        }
    ],
    "locale": "ru",
    "transport": "car",
    "route_mode": "fastest",
    "traffic_mode": "jam"
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print("Error:", response.status_code, response.text)
