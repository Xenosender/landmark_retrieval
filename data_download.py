import os
import csv
import requests
from PIL import Image
from StringIO import StringIO
#from io import BytesIO
from datetime import datetime, time
import sys
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    if key_url_list[0][0] == "id":
        key_url_list.pop(0)
    return key_url_list  # Chop off header


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def find_first_ind(data, prev_downloads, prev_errors):
    current_ind_d = 0
    current_ind_e = 0
    for current_ind, data_line in enumerate(data):
        if current_ind_d < len(prev_downloads) and data_line[0] == prev_downloads[current_ind_d]:
            current_ind_d += 1
        elif current_ind_e < len(prev_errors) and data_line[0] == prev_errors[current_ind_e]:
            current_ind_e += 1
        else:
            if current_ind_d != len(prev_downloads) or current_ind_e != len(prev_errors):
                continue
            else:
                break
    return current_ind


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_example(writer, serialized_image, key, size=None):
    try:
        if size is not None:
            image = Image.open(StringIO(serialized_image)).convert('RGB')
            size_factor = max(image.width, image.height) / float(size)
            image = image.resize([min(size, int(image.width / size_factor)),
                                  min(size, int(image.height / size_factor))])
            new_size = (size, size)
            new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
            new_im.paste(image, ((new_size[0] - image.size[0]) / 2,
                                  (new_size[1] - image.size[1]) / 2))
            image = new_im
            image_io = StringIO()
            image.save(image_io, format='JPEG')
            serialized_image = image_io.getvalue()

        image = Image.open(StringIO(serialized_image)).convert('RGB')
        example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image.size[1]),
                'width': _int64_feature(image.size[0]),
                'key': _bytes_feature(str.encode(key)),
                'img_raw':_bytes_feature(serialized_image)
            }))
        writer.write(example.SerializeToString())
    except Exception as e:
        raise RuntimeError('Error writing {} : {}'.format(key, e.message))


def read_file(tfrecords_filename, read_opts, size):
    current_file_index = 0
    while os.path.exists(tfrecords_filename.format(current_file_index)):
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename.format(current_file_index),
                                                          options=read_opts)

        img_string = ""
        i = 0
        for string_record in record_iterator:

            example = tf.train.Example()
            example.ParseFromString(string_record)

            key = (example.features.feature['key'].bytes_list.value[0])
            height = (example.features.feature['height'].int64_list.value[0])
            width = (example.features.feature['width'].int64_list.value[0])

            img = tf.image.decode_jpeg(example.features.feature['img_raw'].bytes_list.value[0], channels=3)
            img.set_shape([size, size, 3])
            img = tf.image.convert_image_dtype(img, tf.float32)
            yield key, img, width, height

        current_file_index += 1


if __name__ == "__main__":
    CURSOR_UP_ONE = '\033[F'
    ERASE_LINE = '\033[K'

    input_folder = "."
    output_folder = "train"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    nb_images_by_tffile = 1000
    size = 299

    data_file = os.path.join(input_folder, "index.csv")
    key_url_dict = parse_data(data_file)

    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    tfrecords_filename = os.path.join(output_folder, 'img_{}.tfrecords')

    start_time = datetime.now()

    previous_skipped_keys = []
    skipped_file = os.path.join(output_folder, 'skipped.csv')
    if os.path.exists(skipped_file):
        data = parse_data(skipped_file)
        previous_skipped_keys = unique([a[0] for a in data])

    previous_downloaded_keys = []
    downloaded_file = os.path.join(output_folder, 'downloaded.csv')
    if os.path.exists(downloaded_file):
        data = parse_data(downloaded_file)
        previous_downloaded_keys = unique([a[0] for a in data])

    first_ind_to_download = find_first_ind(key_url_dict, previous_downloaded_keys, previous_skipped_keys)

    skipped = open(skipped_file, 'a')
    downloaded = open(downloaded_file, 'a')
    counters = {}
    counters['skipped'] = len(previous_skipped_keys)
    counters['attempts'] = len(previous_skipped_keys) + len(previous_downloaded_keys)

    current_tf_file = 0
    restart_non_zero = False
    while os.path.exists(tfrecords_filename.format(current_tf_file)):
        current_tf_file += 1
        restart_non_zero = True

    writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(current_tf_file), options=opts)
    total = len(key_url_dict)
    print("starting download")
    print('')
    for x in key_url_dict[first_ind_to_download:]:
        key = x[0]
        url = x[1]
        if key in previous_skipped_keys + previous_downloaded_keys:
            continue
        is_skipped = False
        image_data = None
        if counters['attempts'] and counters['attempts'] % nb_images_by_tffile == 0:
            if not restart_non_zero:
                if writer:
                    writer.close()
                current_tf_file += 1
                writer = tf.python_io.TFRecordWriter(tfrecords_filename.format(current_tf_file), options=opts)
            downloaded.flush()
            skipped.flush()
            restart_non_zero = False

        try:
            counters['attempts'] += 1
            response = requests.get(url)
            image_data = response.content

            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line
            print("{:5}% | attempts: {:7} | skipped: {:7} | elapsed: {:8} | {:4} it/s".format(
                round(counters['attempts'] / float(total), 2)
                , counters['attempts']
                , counters['skipped']
                , str((datetime.now() - start_time).seconds).split('.', 2)[0]
                , round(counters['attempts'] / max(1., float((datetime.now() - start_time).seconds)), 2)
            ))
            write_example(writer, image_data, key, size)
            downloaded.write('"{}","{}"\n'.format(key, url))
            restart_non_zero = False
        except Exception as e:
            print(e.message)
            print('')
            skipped.write('"{}","{}"\n'.format(key, url))
            counters['skipped'] += 1
            is_skipped = True

        # if counters['attempts'] > 1:
        #     break

    if downloaded:
        downloaded.close()
    if skipped:
        skipped.close()
    if writer:
        writer.close()


    # data_iterator = read_file(tfrecords_filename, opts, size)
    # with tf.Session() as sess:
    #     for key, img in data_iterator:
    #         img_s = sess.run(img)
    #         plt.imsave(key + ".png", img_s)

