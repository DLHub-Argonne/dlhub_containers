import numpy as np
import tensorflow as tf
import os
from skimage import transform


import data_input
#from convert_to_tfrecords import tags_meta

import resnet_model as model

model_checkpoint_path = './model/model.ckpt-200'

num_tags = 17
gpuid = 0
raw_img_size = 256
model_img_size = 224
batch_size = 1
threhold = 0.5

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph()
sess = tf.Session(config=config, graph=graph)

def load(graph, sess):
    with graph.as_default():
        raw_images_op = tf.placeholder(tf.float32, [batch_size, 256, 256])
        images = tf.expand_dims(raw_images_op, 3)
        labels = tf.placeholder(tf.float32, [batch_size, num_tags])
        # after reading raw images, first resize to fit the model, then normalize the data
        # resize
        images = tf.image.resize_images(images, np.array([model_img_size, model_img_size]))
        # normalize
        std_images = []
        for idx in range(batch_size):
            std_image = tf.image.per_image_standardization(images[idx,:,:,:])
            std_image = tf.expand_dims(std_image, 0)
            std_images.append(std_image)
        images = tf.concat(std_images, 0)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images, is_training=False, num_classes=num_tags)

        # Calculate predictions.
        prob_op = tf.sigmoid(logits)

        # Restore the moving average version of the learned variables for eval.
        saver = tf.train.Saver(tf.global_variables())

        print('load from pretrained model from')
        print(model_checkpoint_path)
        saver.restore(sess, model_checkpoint_path)
        
        return prob_op, raw_images_op

def run(data):
    prob_op, raw_images_op = load(graph, sess)
    
    image = np.log(data) / np.log(1.0414)
    image[np.isinf(image)] = 0
    image = np.expand_dims(image, 0)

    pred_prob = sess.run(prob_op, 
                        feed_dict={raw_images_op: data})
    return pred_prob

def test_run():
    print("transforming image")
    im = np.load('./data/00000003.npy')
    im = transform.resize(im, (256, 256))
    im = im.reshape(1,256,256)
    r = run(im)
    print(r)
