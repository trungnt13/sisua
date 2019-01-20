# add path in order of priority
import os
from odin.utils import select_path

ORIGINAL_BASE_DIR = select_path(
    '/mnt/sdb1/czi_data',
    '/media/data2/czi_data',
    '/data1/czi_data',
create_new=False)

PREPROCESSED_BASE_DIR = select_path(
    '/home/trung/data',
    '/mnt/sdb1/czi_data',
    '/media/data2/czi_data',
    '/data1/czi_data',
create_new=True)

DOWNLOAD_DIR = select_path(
    os.path.join(ORIGINAL_BASE_DIR, 'downloads'),
    create_new=True)

# PATH for saving experiments results
EXP_DIR = select_path('/mnt/sda1/bio_log',
                      create_new=True)
