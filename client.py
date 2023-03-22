import requests

url = 'http://192.168.3.68:5000/replace_json'
new_config = {
    "Devices": [
        {
            "chooseAlgo": [
                {
                    "algoName":"fall_detection",
                    "roi":"1000,1000,1500,1500",
                    "singleAlarmInterval":5,
                    "singleSensitivity":5
                },
                {
                    "algoName":"static_person_detection",
                    "roi":"1000,1000,1500,1500",
                    "singleAlarmInterval":5,
                    "singleSensitivity":5
                }
            ],
            "ipAddress": "10.13.128.13",
            "rtspUrl": "rtsp://admin:admin12345@10.13.128.13:554/Streaming/Channels/101"
        }
    ],
    "DevicesSize": 1
}
headers = {'Content-type': 'application/json'}
response = requests.post(url, json=new_config, headers=headers)

print(response.json()['msg'])
