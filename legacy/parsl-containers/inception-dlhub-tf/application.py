# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import sys
import signal
import codecs

import flask
import tensorflow as tf
import pandas as pd
import base64

#model_path = '/opt/program/models/inception/1'
model_path = './models/inception/1'

def start_sess():
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING], model_path)
    model = sess
    return model

def run(inputs):
    """For the input, do the predictions and return them.
    Args:
        input (a pandas dataframe): The data on which to do the predictions. There will be
            one prediction per row in the dataframe"""
    sess = start_sess()
    if isinstance(inputs, list):
        for i in range(len(inputs)):
            inputs[i] = codecs.decode(inputs[i].encode(), 'base64')
    else:
        inputs = [codecs.decode(inputs.encode(), 'base64')]
    classes = sess.run('classes:0', feed_dict={"image/image:0": inputs})
    scores = sess.run('scores:0', feed_dict={"image/image:0": inputs})
    #print(classes, scores)
    preds = process_output(classes, scores) 
    return preds

def process_output(classes, scores):
    res = []
    for i in range(len(classes[0])):
        res.append( (classes[0][i].decode('utf-8'), str(scores[0][i])) )
        #res[classes[0][i].decode('utf-8')] = str(scores[0][i])
    return res


def test_run():
    """For the input, do the predictions and return them.
    Args:
        input (a pandas dataframe): The data on which to do the predictions. There will be
            one prediction per row in the dataframe"""
    image_path = "Pixiebob-cat.jpg"
    inputs = open(image_path,'rb').read()
    sess = start_sess()
    if not isinstance(inputs, list):
        inputs = [inputs]
    classes = sess.run('classes:0', feed_dict={"image/image:0": inputs})
    scores = sess.run('scores:0', feed_dict={"image/image:0": inputs})
    #print(classes, scores)
    preds = process_output(classes, scores)
    return preds           

if __name__ == "__main__":
    print(test_run())
