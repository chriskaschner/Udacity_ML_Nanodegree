import numpy as np
import tensorflow as tf
import argparse
import json
import requests
from pprint import pprint
import urllib, cStringIO
from PIL import Image
# from urllib import urlretrieve

# imagePath = './Report/images/AltraDifficult01.jpg'
imgURL = 'http://imgdirect.s3-website-us-west-2.amazonaws.com/nike.jpg'
modelFullPath = './Report/output_graph.pb'
labelsFullPath = './Report/output_labels.txt'

# Download the file from `url`, save it in a temporary directory and get the
# path to it (e.g. '/tmp/tmpb48zma.txt') in the `file_name` variable:
imagePath, headers = urllib.urlretrieve(imgURL)

imgFile = cStringIO.StringIO(urllib.urlopen(imgURL).read())
img = Image.open(imgFile)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        print "graph built"


def run_inference_on_image():
    answer = None

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
            print('%s (score = %.5f)' % (human_string, score))

        answer = labels[top_k[0]]
        return answer

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
    print "args are-", args
    # imagePath = args.image_url
    # print "image path is-",imagePath

    run_inference_on_image()

    post_params = {
        "image_url": args.image_url
        # "detect_type": "TEXT_DETECTION"
    }

    # Lazy and used requests in addition to urllib2
    print "post params are- ", post_params
    # r = requests.post(args.endpoint,
    #                   data=json.dumps(post_params),
    #                   headers={'content-type': 'application/json'})
    # detection_results = r.json()
    # pprint(detection_results)
