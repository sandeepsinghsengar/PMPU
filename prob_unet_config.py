# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CityScapes training config."""

import os
import numpy as np
from collections import OrderedDict
config_path = os.path.realpath(__file__)

#########################################
#             data-loader    			#
#########################################

data_dir = '/home/bdp954/PMPU_tf_versio2_org/PMPU_tf_version2/output1/'
resolution = 'quarter'
label_density = 'gtFine'
gt_instances = False
train_cities = ['cities_name']
val_cities = ['cities_name']
num_classes = 8
batch_size = 4
pre_crop_size = [256, 512]
patch_size = [256, 512]
n_train_batches = None
n_val_batches = 274 // batch_size
n_workers = 5
ignore_label = 255

da_kwargs = {
	'random_crop': True,
	'rand_crop_dist': (patch_size[0] / 2., patch_size[1] / 2.),
	'do_elastic_deform': True,
	'alpha': (0., 800.),
	'sigma': (25., 35.),
	'do_rotation': True,
	'angle_x': (-np.pi / 8., np.pi / 8.),
	'angle_y': (0., 0.),
	'angle_z': (0., 0.),
	'do_scale': True,
	'scale': (0.8, 1.2),
	'border_mode_data': 'constant',
	'border_mode_seg': 'constant',
	'border_cval_seg': ignore_label,
	'gamma_retain_stats': True,
	'gamma_range': (0.7, 1.5),
	'p_gamma': 0.3
}

data_format = 'NCHW'
one_hot_labels = False
label_switches=None
#########################################
#          network & training			#
#########################################

cuda_visible_devices = '0'
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'
#gpu_device= gpu_name

network_input_shape = (None, 3) + tuple(patch_size)
network_output_shape = (None, num_classes) + tuple(patch_size)
label_shape = (None, 1) + tuple(patch_size)
loss_mask_shape = label_shape

base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
				6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

num_convs_per_block = 3

#n_training_batches = 240000
n_training_batches = 240
#validation = {'n_batches': n_val_batches, 'every_n_batches': 2000}
validation = {'n_batches': n_val_batches, 'every_n_batches': 200}

learning_rate_schedule = 'piecewise_constant'
#learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
#						'boundaries': [80000, 160000, 240000],
#						'name': 'piecewise_constant_lr_decay'}
learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
						'boundaries': [80, 160, 240],
						'name': 'piecewise_constant_lr_decay'}
initial_learning_rate = learning_rate_kwargs['values'][0]

regularizarion_weight = 1e-5
latent_dim = 6
num_1x1_convs = 3
beta = 1.0
analytic_kl = True
use_posterior_mean = False
#save_every_n_steps = n_training_batches // 3 if n_training_batches >= 100000 else n_training_batches
save_every_n_steps = n_training_batches // 3 if n_training_batches >= 100 else n_training_batches
disable_progress_bar = False
