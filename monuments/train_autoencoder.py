import os
import datetime
import argparse
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.lib.io import file_io
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils


feature_spec = {
    'image': tf.FixedLenFeature([], tf.string),
    'key': tf.FixedLenFeature([1], tf.string),
    'height': tf.FixedLenFeature([1], tf.int64),
    'width': tf.FixedLenFeature([1], tf.int64),
    'image_shape': tf.FixedLenFeature([3], tf.int64)
}


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features=feature_spec)
    image = tf.decode_raw(features['image'], tf.float32)
    key = features['key']
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [8, 8, 1536])
    return (image, height, width, key), features["image_shape"]


def input_fn(filenames, num_epochs=None):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_example)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, label = iterator.get_next()
    return features, label


def generate_model_fn(base_block_size=2,
                      convs_nb_kernels=32,
                      convs_fc_size=612,
                      fc_sizes=[1024, 512, 256, 64],
                      learning_rate=0.1):

    def _model_fn(mode, features, labels):

        if mode == Modes.PREDICT:
            image, height, width, key = features['image'], features['height'], features['width'], features['key']
        else:
            image, height, width, key = features
        blocks = []

        # Extract descriptor blocks
        block_size = base_block_size
        input_shape = image.get_shape().as_list()
        while block_size <= min(input_shape[0], input_shape[1]):
            for i in range(input_shape[0] - block_size + 1):
                for j in range(input_shape[1] - block_size + 1):
                    block = tf.expand_dims(image[i:i+block_size, j:j+block_size, ...], axis=0)
                    if base_block_size < block_size:
                        block = tf.nn.max_pool(block,
                                               [1, block_size/2, block_size/2, 1],
                                               strides=[1, block_size/2, block_size/2, 1],
                                               padding="VALID")
                    blocks.append(block)
            block_size *= 2

        nb_blocks = len(blocks)
        blocks = tf.concat(blocks, axis=0)

        # Apply convs
        conv_sizes = []
        for i in range(1, base_block_size+1):
            for j in range(1, base_block_size+1):
                conv_sizes.append([i, j])

        conv_outputs = []
        for conv_size in conv_sizes:
            with tf.name_scope('encoding_conv_{}_{}'.format(conv_size[0], conv_size[1])):
                weights_conv = tf.Variable(tf.random_normal([conv_size[0],
                                                             conv_size[1],
                                                             input_shape[-1],
                                                             convs_nb_kernels]),
                                            name="conv_weights")
                biases_conv = tf.Variable(tf.zeros([convs_nb_kernels]), name="conv_biases")
                conv_out = tf.nn.conv2d(blocks, weights_conv, strides=[1, 1, 1, 1], padding="VALID")
                conv_out = tf.nn.bias_add(conv_out, biases_conv)
                conv_out = tf.nn.relu(conv_out, name="conv_out")

                weight_fc = tf.Variable(tf.random_normal([2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels, convs_fc_size]),
                                        name="fc_weights")
                biases_fc = tf.Variable(tf.zeros([convs_fc_size]), name='fc_biases')
                conv_out = tf.reshape(conv_out, [-1, 2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels])
                output_fc = tf.add(tf.matmul(conv_out, weight_fc), biases_fc)
                output_fc = tf.nn.relu(output_fc, name="fc_out")

                conv_outputs.append(output_fc)

        output = tf.concat(conv_outputs, axis=1)
        # apply FCs
        for i, fc_size in enumerate(fc_sizes):
            with tf.name_scope('encoding_fc_{}_{}'.format(i, fc_size)):
                last_size = fc_sizes[i-1] if i != 0 else len(conv_sizes) * convs_fc_size
                weight_fc = tf.Variable(tf.random_normal([last_size, fc_size]), name='fc_weights')
                biases_fc = tf.Variable(tf.zeros([fc_size]), name='fc_biases'),
                output = tf.add(tf.matmul(output, weight_fc), biases_fc)
                if i != len(fc_sizes) - 1:
                    output = tf.nn.relu(output, name="fc_out")

        # bottom of the autoencoder : return as descriptors
        if mode == Modes.PREDICT:
            # Convert predicted_indices back into strings
            descriptors = {
                'key': key,
                'descriptors': tf.greater(tf.nn.sigmoid(output), tf.constant(0.5))
            }
            export_outputs = {
                'descriptors': tf.estimator.export.PredictOutput(descriptors)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=descriptors, export_outputs=export_outputs)

        # start decoding

        output = tf.nn.relu(output, name="decoding_in")
        # apply FCs
        dfc_sizes = [len(conv_sizes) * convs_fc_size] + fc_sizes
        for i, fc_size in enumerate(reversed(dfc_sizes)):
            j = len(dfc_sizes) - 1 - i
            with tf.name_scope('decoding_fc_{}_{}'.format(i, fc_size)):
                last_size = dfc_sizes[j + 1] if j != len(dfc_sizes) - 1 else dfc_sizes[-1]
                weight_fc = tf.Variable(tf.random_normal([last_size, fc_size]), name='fc_weights')
                biases_fc = tf.Variable(tf.zeros([fc_size]), name='fc_biases'),
                output = tf.add(tf.matmul(output, weight_fc), biases_fc)
                output = tf.nn.relu(output, name="fc_out")

        dconvs_inputs = [output[:, i*convs_fc_size:(i+1)*convs_fc_size] for i in range(len(conv_sizes))]
        dconv_outputs = []
        for i, conv_size in enumerate(conv_sizes):
            with tf.name_scope('decoding_conv_{}_{}'.format(conv_size[0], conv_size[1])):
                weight_dfc = tf.Variable(tf.random_normal([convs_fc_size, 2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels]),
                                        name="fc_weights")
                biases_dfc = tf.Variable(tf.zeros([2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels]), name='fc_biases')
                output_dfc = tf.add(tf.matmul(dconvs_inputs[i], weight_dfc), biases_dfc)
                output_dfc = tf.nn.relu(output_dfc, name="fc_out")

                deconv_in = tf.reshape(output_dfc, [-1, 2 / conv_size[0], 2 / conv_size[1], convs_nb_kernels])

                weights_deconv = tf.Variable(tf.random_normal([conv_size[0],
                                                             conv_size[1],
                                                             input_shape[-1],
                                                             convs_nb_kernels]),
                                            name="deconv_weights")
                biases_deconv = tf.Variable(tf.zeros([input_shape[-1]]), name="deconv_biases")
                deconv_out = tf.nn.conv2d_transpose(deconv_in, weights_deconv,
                                                    output_shape=[nb_blocks, base_block_size, base_block_size, input_shape[-1]],
                                                    strides=[1, 1, 1, 1],
                                                    padding="VALID")
                deconv_out = tf.nn.bias_add(deconv_out, biases_deconv)
                deconv_out = tf.nn.relu(deconv_out, name="deconv_out")

                dconv_outputs.append(deconv_out)

        final_outputs = tf.add_n(dconv_outputs) / float(len(dconv_outputs))

        loss = tf.losses.mean_squared_error(blocks, final_outputs)

        # Build training operation.
        train_op = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                          l1_regularization_strength=3.0,
                                          l2_regularization_strength=10.0
                                         ).minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[tf.train.LoggingTensorHook([loss], every_n_iter=1)])

    return _model_fn


def serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=None,
        dtype=tf.string
    )
    (image, height, width, key), _ = parse_example(example_bytestring)
    return tf.estimator.export.ServingInputReceiver({"image": image, "width": width, "height": height, "key": key},
                                                    {'example_proto': example_bytestring})


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    file_io.recursive_create_dir(hparams.job_dir)

    train_tfrecords = [os.path.join(hparams.train_dir, f) for f in file_io.list_directory(hparams.train_dir)
                       if 'tfrecords' in f]
    tf.logging.info('Detected {} training files'.format(len(train_tfrecords)))
    train_input = lambda: input_fn(
        train_tfrecords,
        num_epochs=hparams.num_epochs
    )

    # Don't shuffle evaluation data
    eval_tfrecords = [os.path.join(hparams.eval_dir, f) for f in file_io.list_directory(hparams.eval_dir)
                       if 'tfrecords' in f]
    tf.logging.info('Detected {} input files'.format(len(eval_tfrecords)))
    eval_input = lambda: input_fn(eval_tfrecords)

    tf.logging.info('parameters: {}'.format(hparams.to_json()))
    train_spec = tf.estimator.TrainSpec(train_input,
                                        max_steps=hparams.train_steps
                                        )

    exporter = tf.estimator.FinalExporter('auto_encoder', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=hparams.eval_steps,
                                      exporters=[exporter],
                                      name='auto_encoder-eval',
                                      throttle_secs=hparams.throttle_secs)

    model_fn = generate_model_fn(
        base_block_size=hparams.base_block_size,
        convs_nb_kernels=hparams.convs_nb_kernels,
        convs_fc_size=hparams.convs_fc_size,
        fc_sizes=hparams.fc_sizes,
        learning_rate=hparams.learning_rate)

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=hparams.job_dir)
    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-dir',
        help='GCS or local paths to input data',
        required=True
    )
    parser.add_argument(
        '--eval-dir',
        help='GCS or local paths to input data',
        required=True
    )
    parser.add_argument(
        '--output-dir',
        help='GCS location to write checkpoints and export models',
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
    params = {
        "train_dir": args.train_dir,
        "num_epochs": 10,
        "eval_dir": args.eval_dir,
        "train_steps": 1,
        "eval_steps": 1,
        'throttle_secs': 150,
        "base_block_size": 2,
        "convs_nb_kernels": 128,
        "convs_fc_size": 612,
        "fc_sizes": [1024, 512, 256, 64],
        "learning_rate": 0.1,
        "job_dir": os.path.join(args.output_dir, "{}".format(datetime.datetime.now().isoformat()))
    }

    # Run the training job
    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

    hparams=hparam.HParams(**params)
    run_experiment(hparams)