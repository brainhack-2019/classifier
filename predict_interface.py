import requests
import json

'''
A simple interface for sending predictions to the client

e.g.

# When prediction shows gesture with id = 0, then sending is:
res = PredictInterface(0).send()

'''

class PredictInterface(object):
	def __init__(self, gesture_id):
		url = 'https://api.github.com/some/endpoint'
		payload = {
			'gesture_id': gesture_id
		}
	
	def send()
		res = requests.post(
			url,
			data = json.dumps(payload)
		)	

		return res.body

