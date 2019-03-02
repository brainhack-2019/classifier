import requests
import json

def predict_interface(gesture_id):
	url = 'https://api.github.com/some/endpoint'
	payload = {
	'gesture_id': gesture_id
	}
	
	res = requests.post(
		url,
		data = json.dumps(payload)
	)

	print(res.body)

