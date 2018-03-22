import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.training.python.training import hparam


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'key': tf.FixedLenFeature([1], tf.string),
            'height': tf.FixedLenFeature([1], tf.int64),
            'width': tf.FixedLenFeature([1], tf.int64),
            'image_shape': tf.FixedLenFeature([3], tf.int64)
        })
    image = tf.decode_raw(features['image'], tf.float32)
    key = features['key']
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [8, 8, 1536])
    return (image, height, width), key


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

        image, height, width = features
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

        blocks = tf.concat(blocks, axis=0)

        # Apply convs
        conv_sizes = []
        for i in range(1, base_block_size+1):
            for j in range(1, base_block_size+1):
                conv_sizes.append([i, j])

        conv_outputs = []
        for conv_size in conv_sizes:
            with tf.name_scope('conv_{}_{}'.format(conv_size[0], conv_size[1])):
                weights_conv = tf.Variable(tf.random_normal([conv_size[0],
                                                             conv_size[1],
                                                             input_shape[-1],
                                                             convs_nb_kernels]),
                                            name="conv_weights")
                biases_conv = tf.Variable(tf.random_normal([convs_nb_kernels]), name="conv_biases")
                conv_out = tf.nn.conv2d(blocks, weights_conv, strides=[1, 1, 1, 1], padding="VALID")
                conv_out = tf.nn.bias_add(conv_out, biases_conv)
                conv_out = tf.nn.relu(conv_out, name="conv_out")

                weight_fc = tf.Variable(tf.random_normal([2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels, convs_fc_size]),
                                        name="fc_weights")
                biases_fc = tf.Variable(tf.random_normal([convs_fc_size]), name='fc_biases')
                conv_out = tf.reshape(conv_out, [-1, 2 / conv_size[0] * 2 / conv_size[1] * convs_nb_kernels])
                output_fc = tf.add(tf.matmul(conv_out, weight_fc), biases_fc)
                output_fc = tf.nn.relu(output_fc, name="fc_out")

                conv_outputs.append(output_fc)

        output = tf.concat(conv_outputs, axis=1)
        # apply FCs
        for i, fc_size in enumerate(fc_sizes):
            with tf.name_scope('fc_{}_{}'.format(i, fc_size)):
                last_size = fc_sizes[i-1] if i != 0 else len(conv_sizes) * convs_fc_size
                weight_fc = tf.Variable(tf.random_normal([last_size, fc_size]), name='fc_weights')
                biases_fc = tf.Variable(tf.random_normal([fc_size]), name='fc_biases'),
                output = tf.add(tf.matmul(output, weight_fc), biases_fc)
                if i != len(fc_sizes) - 1:
                    output = tf.nn.relu(output, name="fc_out")

        # bottom of the autoencoder : return as descriptors
        if mode == Modes.PREDICT:
            # Convert predicted_indices back into strings
            descriptors = {
              'descriptors': tf.greater(tf.nn.sigmoid(output), tf.constant(0.5))
            }
            export_outputs = {
              'descriptors': tf.estimator.export.PredictOutput(descriptors)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=descriptors, export_outputs=export_outputs)

        # start decoding


        loss = tf.losses.mean_squared_error(image, image)

        # Build training operation.
        train_op = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                          l1_regularization_strength=3.0,
                                          l2_regularization_strength=10.0
                                         ).minimize(loss, global_step=tf.train.global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return _model_fn


def example_serving_input_fn():
    """Build the serving inputs."""
    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    features = parse_example(example_bytestring)
    return tf.estimator.export.ServingInputReceiver(features, {'example_proto': example_bytestring})


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    train_input = lambda: input_fn(
        hparams.train_files,
        num_epochs=hparams.num_epochs
    )

    # Don't shuffle evaluation data
    eval_input = lambda: input_fn(
        hparams.eval_files
    )

    train_spec = tf.estimator.TrainSpec(train_input,
                                        max_steps=hparams.train_steps
                                        )

    exporter = tf.estimator.FinalExporter('auto_encoder',
                                          example_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=hparams.eval_steps,
                                      exporters=[exporter],
                                      name='auto_encoder-eval'
                                      )

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
    tf_directory_train = "train_resnet"
    tfrecords = [os.path.join(tf_directory_train, f) for f in os.listdir(tf_directory_train) if 'tfrecords' in f]

    params = {
        "train_files": tfrecords,
        "num_epochs": 1,
        "eval_files": tfrecords,
        "train_steps": 100,
        "eval_steps": 100,
        "base_block_size": 2,
        "convs_nb_kernels": 128,
        "convs_fc_size": 612,
        "fc_sizes": [1024, 512, 256, 64],
        "learning_rate": 0.1,
        "job_dir": "trainings"
    }

    # Run the training job
    hparams=hparam.HParams(**params)
    run_experiment(hparams)