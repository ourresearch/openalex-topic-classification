try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from io import BytesIO
import json
import boto3
import sys, os, base64, datetime, hashlib, hmac
from chalice import Chalice
from chalice import NotFoundError, BadRequestError

import sys, os, base64, datetime, hashlib, hmac

app = Chalice(app_name='openalex-topic-classifier-v1-endpoint')
app.debug = True

sagemaker = boto3.client('sagemaker-runtime')

@app.route('/', methods=['POST'], content_types=['application/json'], cors=True)
def handle_data():
    # Get the json from the request
    input_json = app.current_request.json_body
    # Send everything to the Sagemaker endpoint
    res = sagemaker.invoke_endpoint(
        EndpointName='openalex-topic-classifier-v1-endpoint',
        Body=input_json,
        ContentType='application/json',
        Accept='Accept'
    )
    return json.loads(res['Body'].read().decode())
