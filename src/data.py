# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 BAIR OPEN RESEARCH COMMONS REPOSITORY
# To view a copy of this license, visit
# https://github.com/wilson1yan/teco/tree/master
# ------------------------------------------------------------------------------

import glob
import os.path as osp
import numpy as np
from flax import jax_utils
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_io as tfio
from tensorflow.python.lib.io import file_io
import io


def is_tfds_folder(path):
    path = osp.join(path, '1.0.0')
    if path.startswith('gs://'):
        return tf.io.gfile.exists(path)
    else:
        return osp.exists(path)


def load_npz(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.npz')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        data = np.load(path)
        video, actions = data['video'].astype(np.float32), data['actions'].astype(np.int32)
        video = 2 * (video / 255.) - 1 
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset


def load_mnist_npz(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*.npz')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        data = np.load(path)
        video, actions = data['data'].astype(np.float32), None
        video = 2 * (video / 255.) - 1
        return video

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.flat_map(
        lambda item: tf.data.Dataset.from_tensor_slices(
            tf.numpy_function(
                read, [item], [tf.float32]
            )[0]
        )
    )
    dataset = dataset.map(
        lambda video: dict(video=video, actions=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


def load_video(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.mp4')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    # TODO resizing video
    def read(path):
        path = path.decode('utf-8')

        # video = tfio.experimental.ffmpeg.decode_video(tf.io.read_file(path)).numpy()
        video = tf.io.read_file(path)
        video = tf.image.decode_video(video)
        video = video.numpy()
        start_idx = np.random.randint(0, video.shape[0] - config.seq_len + 1)
        video = video[start_idx:start_idx + config.seq_len]
        video = 2 * (video / np.array(255., dtype=np.float32)) - 1
        
        np_path = path[:-3] + 'npz'
        if tf.io.gfile.exists(np_path):
            if path.startswith('gs://'):
                np_path = io.BytesIO(file_io.FileIO(np_path, 'rb').read())
            np_data = np.load(np_path)
            actions = np_data['actions'].astype(np.int32)
            actions = actions[start_idx:start_idx + config.seq_len]
        else:
            actions = np.zeros((video.shape[0],), dtype=np.int32)
        
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset

def load_ct(config, split, num_ds_shards, ds_shard_id):
    def convert_ct_binary_to_image(binary_file_path):
        shape = (1024, 1024, 1)
        with open(binary_file_path, 'rb') as fid:
            data = np.fromfile(fid, '>i2')
            image = data.reshape(shape).astype(np.float32)
            return image

    def read_product_path(product_path):
        product_path = product_path.decode('utf-8')
        high_paths = glob.glob(osp.join(product_path, 'high', '*'))
        ct_label = osp.basename(product_path)
        ct_image = []
        for binary_file_path in high_paths:
            image = convert_ct_binary_to_image(binary_file_path)
            if image is not None:
                ct_image.append(image)
        ct_image = np.array(ct_image)
        # 문자열 라벨을 정수 ID로 매핑
        label_id = np.int32(label_to_id[ct_label])
        return ct_image, label_id
    
    product_paths = glob.glob(osp.join(config.data_path, '*'))
    product_paths = [p for p in product_paths if osp.isdir(p)]
    # 문자열 라벨 목록과 매핑 생성 (프로세스/샤드별 동일한 순서 보장 위해 정렬)
    label_names = sorted([osp.basename(p) for p in product_paths])
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    product_paths = np.array_split(product_paths, num_ds_shards)[ds_shard_id].tolist()
        
    dataset = tf.data.Dataset.from_tensor_slices(product_paths)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read_product_path,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    min_ct_len = min(len(ct_image) for ct_image, ct_label in dataset)
    dataset = dataset.map(
        lambda ct_image, ct_label: dict(
            video=2 * ((ct_image[:min_ct_len] - tf.reduce_min(ct_image[:min_ct_len])) / (tf.reduce_max(ct_image[:min_ct_len]) - tf.reduce_min(ct_image[:min_ct_len]))) - 1,
            actions=None,
            label=ct_label
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset

def load_ct_slices(config, split, num_ds_shards, ds_shard_id):
    def convert_ct_binary_to_image(binary_file_path):
        shape = (1024, 1024, 1)
        with open(binary_file_path, 'rb') as fid:
            data = np.fromfile(fid, '>i2')
            image = data.reshape(shape).astype(np.float32)
            return image

    def read_product_paths(product_paths):
        ct_slices = []
        ct_slices_order = []
        for product_path in product_paths:
            high_paths = glob.glob(osp.join(product_path, 'high', '*'))
            high_paths.sort(key=lambda x: osp.basename(x))
            ct_image = []
            for binary_file_path in high_paths:
                image = convert_ct_binary_to_image(binary_file_path)
                if image is not None:
                    ct_image.append(image)
            ct_image = np.array(ct_image)
            # slicing for every 10 frames
            # for i in range(0, len(ct_image)):
            #     select_idx = i + np.arange(0,config.slice_interval*config.slice_seq_len, config.slice_interval)
            #     if select_idx[-1] >= len(ct_image):
            #         break
            #     ct_slice = ct_image[select_idx]
            for i in range(0, len(ct_image), config.slice_interval):
                if i + config.slice_seq_len > len(ct_image):
                    break
                ct_slice = ct_image[i:i+config.slice_seq_len]
                # shuffle ct_slice and save the order to recover the original slice to ct_slice_order
                for _ in range(10): # data augmentation
                    idxs = np.arange(1,len(ct_slice))
                    np.random.shuffle(idxs)
                    idxs = np.concatenate([[0], idxs])
                    perm = np.argsort(idxs)
                    ct_slices.append(ct_slice[idxs])
                    ct_slices_order.append(perm)

        ct_slices = np.array(ct_slices, dtype=np.float32) / 150.
        ct_slices_order = np.array(ct_slices_order, dtype=np.int32)
        # ct_slices_order is (N, T) -> one_hot_labels is (N, T, T)
        one_hot_labels = np.eye(config.slice_seq_len, dtype=np.int32)[ct_slices_order]
        return ct_slices, one_hot_labels
    
    product_paths = glob.glob(osp.join(config.data_path, '*'))
    product_paths = [p for p in product_paths if osp.isdir(p)]
    product_paths = np.array_split(product_paths, num_ds_shards)[ds_shard_id].tolist()

    ct_slices, ct_slices_order = read_product_paths(product_paths)

    # shuffle slice_indices with a fixed key
    slice_indices = np.arange(0, len(ct_slices))
    key = jax.random.PRNGKey(config.seed)
    slice_indices = jax.random.shuffle(key, slice_indices)
    if split == 'train':
        slice_indices = slice_indices[:int(len(slice_indices) * 0.9)]
    elif split == 'test':
        slice_indices = slice_indices[int(len(slice_indices) * 0.9):]
    ct_slices = ct_slices[slice_indices]
    ct_slices_order = ct_slices_order[slice_indices]

    print(f'{ct_slices.shape[0]} slices, {ct_slices.shape[1]} frames per slice')
    dataset = tf.data.Dataset.from_tensor_slices((ct_slices, ct_slices_order))
    dataset = dataset.map(
        lambda ct_slice, ct_slice_order: (tf.cast(ct_slice, tf.float32), tf.cast(ct_slice_order, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda ct_slice, ct_slice_order: dict(
            video=ct_slice,
            actions=None,
            labels=ct_slice_order
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset

class Data:
    def __init__(self, config, xmap=False):
        self.config = config
        self.xmap = xmap
        print('Dataset:', config.data_path)

    @property
    def train_itr_per_epoch(self):
        return self.train_size // self.config.batch_size

    @property
    def test_itr_per_epoch(self):
        return self.test_size // self.config.batch_size

    def create_iterator(self, train, repeat=True, prefetch=True):
        if self.xmap:
            num_data = jax.device_count() // self.config.num_shards
            num_data_local = max(1, jax.local_device_count() // self.config.num_shards)
            if num_data >= jax.process_count():
                num_ds_shards = jax.process_count()
                ds_shard_id = jax.process_index()
            else:
                num_ds_shards = num_data
                n_proc_per_shard = jax.process_count() // num_data
                ds_shard_id = jax.process_index() // n_proc_per_shard
        else:
            num_data_local = jax.local_device_count()
            num_ds_shards = jax.process_count()
            ds_shard_id = jax.process_index()

        batch_size = self.config.batch_size // num_ds_shards
        split_name = 'train' if train else 'test'

        if not is_tfds_folder(self.config.data_path):
            if 'KITECH' in self.config.data_path:
                dataset = load_ct_slices(self.config, split_name, num_ds_shards, ds_shard_id)
                # dataset = load_ct(self.config, split_name, num_ds_shards, ds_shard_id)
            elif 'dmlab' in self.config.data_path:
                dataset = load_npz(self.config, split_name, num_ds_shards, ds_shard_id)
            elif 'mnist' in self.config.data_path:
                dataset = load_mnist_npz(self.config, split_name, num_ds_shards, ds_shard_id)
            else:
                dataset = load_video(self.config, split_name, num_ds_shards, ds_shard_id)
        else:
            seq_len = self.config.seq_len

            def process(features):
                video = tf.cast(features['video'], tf.int32)
                T = tf.shape(video)[0]
                start_idx = tf.random.uniform((), 0, T - seq_len + 1, dtype=tf.int32)
                video = tf.identity(video[start_idx:start_idx + seq_len])
                actions = tf.cast(features['actions'], tf.int32)
                actions = tf.identity(actions[start_idx:start_idx + seq_len])
                return dict(video=video, actions=actions)

            split = tfds.split_for_jax_process(split_name, process_index=ds_shard_id,
                                               process_count=num_ds_shards)
            dataset = tfds.load(osp.basename(self.config.data_path), split=split,
                                data_dir=osp.dirname(self.config.data_path))

            # caching only for pre-encoded since raw video will probably
            # run OOM on RAM
            if self.config.cache:
                dataset = dataset.cache()

            options = tf.data.Options()
            options.threading.private_threadpool_size = 48
            options.threading.max_intra_op_parallelism = 1
            dataset = dataset.with_options(options)
            dataset = dataset.map(process)

        if repeat:
            dataset = dataset.repeat()
        if train:
            dataset = dataset.shuffle(batch_size * 32, seed=self.config.seed)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)

        def prepare_tf_data(xs):
            def _prepare(x):
                x = x._numpy()
                x = x.reshape((num_data_local, -1) + x.shape[1:])
                return x
            xs = jax.tree_util.tree_map(_prepare, xs)
            return xs

        iterator = map(prepare_tf_data, dataset)

        if prefetch:
            iterator = jax_utils.prefetch_to_device(iterator, 2)

        return iterator
