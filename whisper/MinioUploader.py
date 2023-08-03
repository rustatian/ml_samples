from minio import Minio
from minio.error import S3Error

class MinioUploader:
	def __init__(self, endpoint, access_key, secret_key, secure=True):
		self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
	
	def upload(self, bucket_name, object_name, file_path):
		try:
			result = self.client.fput_object(bucket_name, object_name, file_path)
			print("uploaded", result)
		except S3Error as exc:
			print("error: ", exc)
