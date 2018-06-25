# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging
import datetime

from flask import Flask, send_from_directory
from flask.json import jsonify
from flask import request
import json
import requests
import os
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy
import base64
from google.cloud import storage
import six
from werkzeug import secure_filename
from werkzeug.exceptions import BadRequest
import tempfile
from PIL import Image

from io import BytesIO
from urllib.parse import unquote
from flask_cors import CORS

    #production WSGI webserver
import cherrypy
from paste.translogger import TransLogger

from sklearn.externals import joblib
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.models import Model
from keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__, static_folder='build')
app.config.from_object('config')
cors = CORS(app, resources={r"/*": {"origins": "*"}})



# Environment variables are defined in app.yaml.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['SQLALCHEMY_DATABASE_URI']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

PUBLIC_PREFIX = "https://storage.googleapis.com/"

#Load Image ID Labels
image_ids = np.load('models/image_ids.pkl')

#Load Feature Extractor
global graph 
graph = tf.get_default_graph()
model = InceptionV3(weights='imagenet')
bottleneck_model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)

#Load PCA or Autoencoder Features
auto_encoder = load_model('models/all_encoder.h5')
principleComponents = joblib.load('models/pca_transformer.pkl')

#Load KD-Trees
tree_pca = joblib.load('models/tree_pca.pkl')
tree_auto = joblib.load('models/tree_auto.pkl')

class LabelledWardrobe(db.Model):
    # __table__ = db.Model.metadata.tables['labelled_wardrobe']
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(255))
    image_url = db.Column(db.String(255))
    labels = db.Column(db.String(255))

    def __init__(self, user_id, image_url, labels):
        self.user_id = user_id
        self.image_url = image_url
        self.labels = labels

def from_sql(row):
    """Translates a SQLAlchemy model instance into a dictionary"""
    data = row.__dict__.copy()
    return (data['user_id'], data['image_url'], data['labels'])        
        
def _get_storage_client():
    return storage.Client(
        project=app.config['PROJECT_ID'])

def base64ToImage(encoded_data):
    data = base64.b64decode(encoded_data)
    return data

def _check_extension(filename, allowed_extensions):
    if ('.' not in filename or
            filename.split('.').pop().lower() not in allowed_extensions):
        raise BadRequest(
            "{0} has an invalid name or extension".format(filename))        

def _safe_filename(filename):
    """
    Generates a safe filename that is unlikely to collide with existing objects
    in Google Cloud Storage.
    ``filename.ext`` is transformed into ``filename-YYYY-MM-DD-HHMMSS.ext``
    """
    filename = secure_filename(filename)
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    basename, extension = filename.rsplit('.', 1)
    return "{0}-{1}.{2}".format(basename, date, extension)

# [START upload_file]
def upload_file(user_id, file_stream, filename, content_type):
    """
    Uploads a file to a given Cloud Storage bucket and returns the public url
    to the new object.
    """
    _check_extension(filename, app.config['ALLOWED_EXTENSIONS'])
    filename = _safe_filename(filename)

    client = _get_storage_client()
    bucket = client.bucket(app.config['CLOUD_STORAGE_BUCKET'])
    blob = bucket.blob("user_images/"+user_id+"/"+filename)

    blob.upload_from_string(
        file_stream,
        content_type=content_type)

    url = blob.public_url

    if isinstance(url, six.binary_type):
        url = url.decode('utf-8')
    return url
# [END upload_file]


def list_filenames(user_id):
    """Lists all the blobs in the bucket."""
    prefix = "user_images/"+user_id+"/"
    url = "https://www.googleapis.com/storage/v1/b/flippers_gcp/o/?prefix="+prefix
    response = json.loads(requests.get(url).text)
    image_urls = []
    for resp in response['items']:
        image_urls.append(resp['name'])
    return image_urls

def image_detection(gcs_image_uri):
    DETECTION_TYPE = 'LABEL_DETECTION'
    keys = {"key": "AIzaSyAYqWz0NkVI69pZL6Em_xxRVjKohMQx5lc"}
    payload = {
        'requests':{
            'features':[
                {
                    'type':DETECTION_TYPE,
                    'maxResults':20
                }
            ],
            'image':{
                "source":{
                    "imageUri":
                        gcs_image_uri
                    }
                }
            }
        }    
    headers = {'content-type': 'application/json'}
    response = json.loads(requests.request(
        "POST",
        "https://vision.googleapis.com/v1/images:annotate",
        data = json.dumps(payload), headers=headers, params = keys
    ).text)
    if 'error' in response:
        raise Exception("Google Vision Threw an Error -> {}".format(response))
    else:
        all_labels = []
        if 'labelAnnotations' in response['responses'][0]:
            labels = response['responses'][0]['labelAnnotations']
            for l in labels:
                all_labels.append(l['description'])
        return ','.join(all_labels)        


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists("build/" + path):
        return send_from_directory('build', path)
    else:
        return send_from_directory('build', 'index.html')


@app.route('/ping')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello ping!'

@app.route('/label_images')
def vision():
    # UserID
    user_id = request.args.get("uid")
    # All images in a given bucket per user
    object_urls = list_filenames(user_id)
    bucket_id = "flippers_gcp"
    for url in object_urls:
        gcs_image_uri = 'gs://{}/{}'.format(bucket_id, url)
        labels = image_detection(gcs_image_uri)
        labelled_image = LabelledWardrobe(
            user_id=user_id,
            image_url=gcs_image_uri,
            labels=labels
        )
        db.session.merge(labelled_image)
        db.session.commit()
    return 'user_id:'+ user_id+'labels:'+labels


@app.route('/upload_wardrobe', methods=["POST"])
def upload_wardrobe():
    """Process the uploaded file and upload it to Google Cloud Storage."""
    req = request.get_json()
    image = req["image"]
    user_id = req["uid"]
    decoded_image = base64ToImage(image)
    filename = tempfile.NamedTemporaryFile().name
    content_type = 'image/jpeg'
    gcs_image_uri =  unquote(unquote(upload_file(user_id, decoded_image, filename+'.jpeg', content_type).replace(PUBLIC_PREFIX,"gs://")))
    labels = image_detection(gcs_image_uri)
    labelled_image = LabelledWardrobe(
            user_id=user_id,
            image_url=gcs_image_uri,
            labels=labels
        )
    db.session.add(labelled_image)
    db.session.commit()
    return jsonify({"response":200})

@app.route('/view_wardrobe')
def view_wardrobe():
    """View the wardrobe for a given user id"""
    final_response = []
    user_id = request.args.get("uid")
    # result = LabelledWardrobe.query.first()
    results = db.session.query(LabelledWardrobe).filter_by(user_id = user_id).all()
    if not results:
        return jsonify(final_response)
    for result in results:
        user_id, image_url, labels = from_sql(result)
        image_url = image_url.replace("gs://",PUBLIC_PREFIX)
        response = {'image_url':image_url, 'labels':labels}
        final_response.append(response)    
    return jsonify(final_response)

@app.route('/delete')
def delete_image():
    user_id = request.args.get("uid")
    image_url = request.args.get("image_url").replace(PUBLIC_PREFIX,"gs://")
    db.session.query(LabelledWardrobe).filter_by(user_id = user_id, image_url = image_url).delete()
    db.session.commit()
    return "hi"    

@app.route('/search')
def search():
    """Return image corresponding to text search"""
    return 'Search'

def return_recommended_images(input_image, amount):
    img = input_image.resize((299, 299), Image.ANTIALIAS)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)[:,:,:,0:3]

    with graph.as_default():
        sanjana_features = bottleneck_model.predict(x)
        reduced_sanjana_auto = auto_encoder.predict(sanjana_features) 
    reduced_sanjana_pca = principleComponents.transform(sanjana_features)

    k = amount
    dist_pca, ind_pca = tree_pca.query([reduced_sanjana_pca[0]], k=k) 
    dist_auto, ind_auto = tree_auto.query([reduced_sanjana_auto[0]], k=k) 

    return [str(int(f)) for f in list(image_ids[ind_pca][0])]

@app.route('/recommended_images', methods=["POST"])
def recommended_images():
    """Returns recommended images based on a users wardrobe"""
    final_response = []
    req = request.get_json()
    img_src = req["image"]
    amount = req["amount"]
    # decoded_image = Image.open(BytesIO(base64ToImage(image))) 
    image_data = Image.open(BytesIO(requests.get(img_src).content))
    image_ids = return_recommended_images(image_data, amount)
    for id in image_ids:
        url = PUBLIC_PREFIX+"flippers_gcp/all_products/"+id+".jpg"
        final_response.append(url)
    return jsonify({"response":final_response}) 
 
@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    # app.run(host='0.0.0.0', port=8080, debug=True)

    print("Starting Server")
    app_logged = TransLogger(app)

    cherrypy.tree.graft(app_logged, '/')
    cherrypy.config.update({
        'global':{
            'engine.autoreload_on': True,
            'log.screen': True,
            'server.socket_port': 8080,
            'server.socket_host': '0.0.0.0',
            'server.ssl_module':'builtin',
            'server.ssl_certificate':'cert.pem',
            'server.ssl_private_key':'privkey.pem'
        }
    })

    cherrypy.engine.start()
    cherrypy.engine.block()
# [END app]