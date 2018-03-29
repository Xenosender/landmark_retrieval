import os
import csv
import requests
from datetime import datetime
from PIL import Image
from StringIO import StringIO
import sys
import multiprocessing


def parse_data(data_file, already_downloaded):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader if line[0] not in already_downloaded]
    return key_url_list[1:]  # Chop off header


if __name__ == "__main__":
    CURSOR_UP_ONE = '\033[F'
    ERASE_LINE = '\033[K'

    input_folder = "."
    output_folder = "raw"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    already_downloaded = [f[:f.rfind('.')] for f in os.listdir(output_folder) if 'jpg' in f]

    data_file = os.path.join(input_folder, "index.csv")
    key_url_list = parse_data(data_file, already_downloaded)

    start_time = datetime.now()

    previous_skipped_keys = []
    skipped_file = os.path.join(output_folder, 'skipped.csv')
    output_file = os.path.join(output_folder, '{}.{}')
    start_ind = len(already_downloaded)

    def download_url(ind_key_url):
        ind, key_url = ind_key_url
        key = key_url[0]
        url = key_url[1]
        try:
            response = requests.get(url)
            image_data = response.content
            image = Image.open(StringIO(image_data)).convert('RGB')
            image.save(output_file.format(key, 'jpg'))

            lock.acquire()
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line
            sys.stdout.write("{}\n".format(start_ind + ind))
            lock.release()
            return key
        except Exception as e:
            print(e.message)
            print('')
            lock.acquire()
            with open(skipped_file, 'a') as f:
                f.write('"{}","{}"\n'.format(key, url))
            lock.release()
            return 'error'

    file_lock = multiprocessing.Lock()

    def init(l):
        global lock
        lock = l

    pool = multiprocessing.Pool(processes=10, initializer=init, initargs=(file_lock,))

    total = len(key_url_list)
    print("starting download")
    print('')
    pool.map(download_url, enumerate(key_url_list))
    pool.close()
    pool.join()

