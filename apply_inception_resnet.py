import argparse
import os

import tensorflow as tf
import numpy as np
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from data_download import read_file
from tensorflow.contrib.training.python.training import hparam


def model_fn(features):
    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
        output = inception_resnet_v2(features, num_classes=None, is_training=False,
                                     dropout_keep_prob=1,
                                     reuse=None,
                                     scope='InceptionResnetV2',
                                     create_aux_logits=False,
                                     activation_fn=tf.nn.relu)

    tf.train.init_from_checkpoint(
        'inception_resnet_v2_2016_08_30.ckpt',
        {"InceptionResnetV2/": "InceptionResnetV2/"})

    return output


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_example(writer, image, key, height, width):
    try:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'key': _bytes_feature(str.encode(key)),
                    'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'image_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(image.shape))),
                }))
        writer.write(example.SerializeToString())
    except Exception as e:
        raise RuntimeError('Error writing {} : {}'.format(key, e.message))


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    read_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    input_size = 299
    if not os.path.exists(hparams.output_dir):
        os.makedirs(hparams.output_dir)
    output_file = os.path.join(hparams.output_dir, 'inception_tfrecord_{}.tfrecords')
    current_output_file = hparams.start_index_output
    nb_output_by_file = 10000
    current_output_ind = 0

    writer = tf.python_io.TFRecordWriter(output_file.format(current_output_file))
    error_files = os.path.join(hparams.output_dir, 'errors.csv')
    errors = open(error_files, 'a')

    with tf.Graph().as_default():
        input_placeholder = tf.placeholder(tf.float32, shape=[1, 299, 299, 3])
        logits, endpoints = model_fn(input_placeholder)
        last_conv_endpoint = endpoints["Conv2d_7b_1x1"]

        with tf.Session() as sess:
            init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
            sess.run(init)

            for i in range(hparams.nb_files):
                input_tfrecord = hparams.input_files.format(i)
                if not os.path.exists(input_tfrecord):
                    print('{} not found : quitting'.format(input_tfrecord))
                    return

                data_iterator = read_file(input_tfrecord, read_opts, input_size)

                for key, img, width, height in data_iterator:
                    if current_output_ind and current_output_ind % nb_output_by_file == 0:
                        if writer:
                            writer.close()
                        current_output_file += 1
                        writer = tf.python_io.TFRecordWriter(output_file.format(current_output_file))
                        errors.flush()

                    res_in = sess.run(img)
                    res_in = res_in[np.newaxis, ...]
                    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
                        # 8, 8, 1536, float32
                        conv_out = sess.run(last_conv_endpoint, feed_dict={input_placeholder: res_in})
                    try:
                        write_example(writer, conv_out[0], key, height, width)
                        current_output_ind += 1

                    except:
                        print('{} not converted'.format(key))
                        errors.write(key + "\n")

    if errors:
        errors.close()
    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--input-files',
        help='GCS or local paths to input data',
        required=True
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0
    )
    parser.add_argument(
        '--nb-files',
        type=int,
        required=True
    )
    parser.add_argument(
        '--output-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--start-index-output',
        type=int,
        default=0
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
        help='Set logging verbosity'
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)