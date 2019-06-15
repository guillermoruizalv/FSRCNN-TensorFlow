import os
from model import Model

import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("arch", "FSRCNN", "Model name [FSRCNN]")
flags.DEFINE_boolean("fast", False, "Use the fast model (FSRCNN-s) [False]")
flags.DEFINE_integer("epoch", 10, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of the adam optimizer [1e-4]")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [2]")
flags.DEFINE_integer("radius", 1, "Max radius of the deconvolution input tensor [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("data_dir", "Train", "Name of data directory to train on [Train]")
flags.DEFINE_string("test_dir", "Test", "Name of data directory to test [Test]")
flags.DEFINE_string("val_dir", "Validation", "Name of data directory to use for validation [Validation]")
flags.DEFINE_string("output_dir", "results", "Name of test output directory [results]")
flags.DEFINE_boolean("train", True, "True for training, false for testing [True]")
flags.DEFINE_integer("threads", 1, "Number of processes to pre-process data with [1]")
flags.DEFINE_boolean("distort", False, "Distort some images with JPEG compression artifacts after downscaling [False]")
flags.DEFINE_boolean("params", False, "Save weight and bias parameters [False]")

FLAGS = flags.FLAGS

def main(_):

  if FLAGS.fast:
    FLAGS.checkpoint_dir = 'fast_{}'.format(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  with tf.Session() as sess:
    model = Model(sess, config=FLAGS)
    model.run()
    
if __name__ == '__main__':
  tf.app.run()
