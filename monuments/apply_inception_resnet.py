import argparse
import os

import tensorflow as tf
import numpy as np
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.contrib.training.python.training import hparam

from tensorflow.python.lib.io import file_io


def parse_example(serialized_example, img_size):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'key': tf.FixedLenFeature([1], tf.string),
            'height': tf.FixedLenFeature([1], tf.int64),
            'width': tf.FixedLenFeature([1], tf.int64)
        })

    key = features['key']
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.image.decode_jpeg(features['img_raw'], channels=3)
    image.set_shape([img_size, img_size, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return key, image, height, width


def input_fn(filenames, img_size, read_opts):
    dataset = tf.data.TFRecordDataset(filenames, compression_type=read_opts)
    dataset = dataset.map(lambda x: parse_example(x, img_size))
    dataset = dataset.batch(10)
    iterator = dataset.make_one_shot_iterator()
    key, image, height, width = iterator.get_next()
    return key, image, height, width


def model_fn(features, inception_dir):
    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
        output = inception_resnet_v2(features, num_classes=None, is_training=False,
                                     dropout_keep_prob=1,
                                     reuse=None,
                                     scope='InceptionResnetV2',
                                     create_aux_logits=False,
                                     activation_fn=tf.nn.relu)

    tf.train.init_from_checkpoint(
        os.path.join(inception_dir, 'inception_resnet_v2_2016_08_30.ckpt'),
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

    read_opts = "ZLIB"
    input_size = 299
    if not file_io.file_exists(hparams.output_dir):
        file_io.recursive_create_dir(hparams.output_dir)
    output_file = os.path.join(hparams.output_dir, 'inception_tfrecord_{}.tfrecords')
    current_output_file = hparams.start_index_output
    nb_output_by_file = 10000
    current_output_ind = 0

    writer = tf.python_io.TFRecordWriter(output_file.format(current_output_file))
    error_files = os.path.join(hparams.output_dir, 'errors.csv')
    errors = file_io.FileIO(error_files, 'a')

    input_files = [f for f in file_io.list_directory(hparams.input_dir) if 'tfrecord' in f]
    if not len(input_files):
        tf.logging.error('Error: no file detected')
        return
    tf.logging.info('Detected {} input files'.format(len(input_files)))
    input_files = sorted(input_files, key=lambda f:int(f[f.rfind('_')+1:f.rfind('.')]))
    input_files = [os.path.join(hparams.input_dir, f) for f in input_files]

    with tf.device("/device:GPU:0"):
        key, image, height, width = input_fn(input_files, input_size, read_opts)
        logits, endpoints = model_fn(image, hparams.inception_dir)
        last_conv_endpoint = endpoints["Conv2d_7b_1x1"]

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
            sess.run(init)
            sess.graph.finalize()

            while True:
                try:
                    # tf.logging.info('running doc {}'.format(key))
                    if current_output_ind and current_output_ind % nb_output_by_file == 0:
                        tf.logging.info('closing writer {}'.format(current_output_file))
                        if writer:
                            writer.close()
                        current_output_file += 1
                        writer = tf.python_io.TFRecordWriter(output_file.format(current_output_file))
                        errors.flush()

                    with tf.contrib.slim.arg_scope(inception_resnet_v2_arg_scope()):
                        # 8, 8, 1536, float32
                        key_val, h, w, conv_out = sess.run([key, height, width, last_conv_endpoint])

                        for k in range(key_val.shape[0]):
                            try:
                                write_example(writer, conv_out[k], key_val[k][0], h[k][0], w[k][0])
                                current_output_ind += 1
                            except Exception as e:
                                tf.logging.warning('{} not converted : {}'.format(key_val[k][0], e.message))
                                errors.write(key_val[k][0] + "\n")

                    if current_output_ind and current_output_ind % 1000 == 0:
                        tf.logging.info('Processed {}'.format(current_output_ind))
                except tf.errors.OutOfRangeError:
                    break
                # tf.logging.info('{} found in file {}'.format(nb_examples_in_file, input_f))

    if errors:
        errors.close()
    if writer:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--input-dir',
        help='GCS or local paths to input data',
        required=True
    )
    parser.add_argument(
        '--start-index',
        type=int,
        default=0
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
        '--inception-dir',
        required=True
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

    args, unknown = parser.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams=hparam.HParams(**args.__dict__)
    run_experiment(hparams)