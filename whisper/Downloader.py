import requests

class Downloader:
	def __init__(self, url):
		self.url = url

	def download(self):
		response = requests.get(self.url, Stream=True)
		if response.status_code == 200:
			return response.content