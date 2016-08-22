import numpy as np
import tensorflow as tf
import argparse
import json
import requests
from pprint import pprint
import urllib, cStringIO
from PIL import Image
# from urllib import urlretrieve
from flask import Flask, jsonify, abort, make_response, request

app = Flask(__name__)

# datastore for this example; ###todo - add database
images = [
    {
        'id': 1,
        'title': u'Nikes',
        'url': 'http://imgdirect.s3-website-us-west-2.amazonaws.com/nike.jpg'
    },
    {
        'id': 2,
        'title': u'Altra',
        'url': 'http://imgdirect.s3-website-us-west-2.amazonaws.com/altra.jpg'
    }
]
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/img/api/v1.0/images', methods=['GET'])
def get_images():
    return jsonify({'images': images})

@app.route('/img/api/v1.0/images/<int:img_id>', methods=['GET'])
def get_image(img_id):
    img = [img for img in images if img['id'] == img_id]
    if len(img) == 0:
        abort(404)
    return jsonify({'img': img[0]})

### test String
### curl -i -H "Content-Type: application/json" -X POST -d '{"url":"http://imgdirect.s3-website-us-west-2.amazonaws.com/neither.jpg"}' http://127.0.0.1:5000/img/api/v1.0/images
@app.route('/img/api/v1.0/images', methods=['POST'])
def create_task():
    if not request.json or not 'url' in request.json:
        abort(400)

    image = {
        ### simple way to ensure a unique id, just add 1
        'id' : images[-1]['id'] + 1,
        ### allow an empty title
        'title': request.json.get('title', ""),
        ### url is required, otherwise return error code 400
        'url': request.json['url'],
        ###todo add logic to retrieve results from image prediction
        'results': run_inference_on_image(request.json['url'])

    }
    images.append(image)
    return jsonify({'image': image}), 201


# imagePath = './Report/images/AltraDifficult01.jpg'
# imgURL = 'http://imgdirect.s3-website-us-west-2.amazonaws.com/nike.jpg'
modelFullPath = './output_graph.pb'
labelsFullPath = './output_labels.txt'
results_name = []
results_score = []
results = []

# Download the file from `url`, save it in a temporary directory and get the
# path to it (e.g. '/tmp/tmpb48zma.txt') in the `file_name` variable:
# imagePath, headers = urllib.urlretrieve(imgURL)
#
# imgFile = cStringIO.StringIO(urllib.urlopen(imgURL).read())
# img = Image.open(imgFile)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(imgURL):
    answer = None
    imagePath, headers = urllib.urlretrieve(imgURL)
    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            results_name.append(human_string)
            results_score.append(score)
            print('%s (score = %.5f)' % (human_string, score))
        # answer = labels[top_k[0]]
        results = zip(results_name, results_score)
        results_dict = {
            "results_name_1": results_name[0],
            "results_score_1": json.JSONEncoder().encode(format(results_score[0], '.5f')),
            "results_name_2": results_name[1],
            "results_score_2": json.JSONEncoder().encode(format(results_score[1], '.5f')),
            "results_name_3": results_name[2],
            "results_score_3": json.JSONEncoder().encode(format(results_score[2], '.5f'))
        }
        return results_dict

## should use "try... "
# def is_valid_file(parser, arg):
#     try:
#
#     if not os.path.exists(arg):
#         parser.error("The file %s does not exist!" % arg)
#     else:
#         return open(arg, 'r')  # return an open file handle

# if __name__ == '__main__':
#     run_inference_on_image()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="analyzes image for brand logos"
    )
    parser.add_argument(
        "-i", "--image_url",
        help="The image URL to process",
        required=True
    )
    parser.add_argument(
        "-e", "--endpoint",
        help="The API Gateway endpoint to use",
        required=False,
        default='https://596kyescv8.execute-api.us-east-1.amazonaws.com/dev/detect_image'
    )
# # http://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
# # path to graph file
#         "-g", "--graph",
#         help="path to graph file to use in TensorFlow, Default: %(default)s",
#         default='./output_graph.pb',
#         type=int
#     )

    args = parser.parse_args()

    # imagePath = args.image_url
    # print "image path is-",imagePath
    imgURL = args.image_url
    imagePath, headers = urllib.urlretrieve(imgURL)
    # run_inference_on_image()

    # post_params = {
        # "image_url": args.image_url,
        # "results_name_1": results_name[0],
        # "results_score_1": json.JSONEncoder().encode(format(results_score[0], '.5f')),
        # "results_name_2": results_name[1],
        # "results_score_2": json.JSONEncoder().encode(format(results_score[1], '.5f')),
        # "results_name_3": results_name[2],
        # "results_score_3": json.JSONEncoder().encode(format(results_score[2], '.5f'))
    # }
    #
    # # Lazy and used requests in addition to urllib2
    # print "post params are- ", post_params
    # print "results are: ", results
    # r = requests.post(args.endpoint,
    #                   data=json.dumps(post_params),
    #                   headers={'content-type': 'application/json'})
    # detection_results = r.json()
    # pprint(detection_results)
    app.run(debug=True)
