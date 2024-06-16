import torch
import logging


save_model_dir = '/xxxx/checkpoints/ssd'
train_root_dir = '/xxxx/spp32768_train'
val_root_dir = '/xxxx/spp32768_val'
test_image_dir = '/xxxx/spp32768_test'
save_image_dir = '/xxxx/checkpoints/ssd'

model_name = 'bistro_temporal_base'

recursion_step = 5
epoch_num = 200
tb_record_interval = 10

# ------- param --------------
dilation = False
deform = False
# -----------------------------

train_width = 256
test_width = 512
batch_size = 8
num_workers = 20
best_checkpoint = True

resume = False
resume_model_path = ''
train_scenes = ['Bistro']
test_scenes = ['Bistro']
# spp0.25 is named as images
train_default_size = [1024, 2048]
test_default_size = [1024, 2048]
train_spp_dir = 'images'
test_spp_dir = 'images'
cache = True
data_repeat = 10
multi_gpu = False # torch.cuda.device_count() > 1
loss_weight = [0.05, 0.25, 0.5, 0.75, 1]

manual_random_seed = 42

# -----------logger--------------
logger = logging.getLogger('train')
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# ----------- Val logger--------------
val_logger = logging.getLogger('val')
val_logger.setLevel(logging.DEBUG)
