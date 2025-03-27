import os.path
import torch
from torch.utils.data import ConcatDataset, Dataset
import scipy.io as sio
import hdf5storage
from scipy import sparse
import numpy as np
import sklearn.preprocessing as skp


def load_mat(args):
    data_X = []
    label_y = None


    if args.dataset == 'Scene15':
        mat = sio.loadmat(os.path.join(args.data_path, 'Scene-15.mat'))
        X = mat['X'][0]
        data_X.append(X[0].astype('float32'))
        data_X.append(X[1].astype('float32'))
        label_y = np.squeeze(mat['Y'])

    elif args.dataset == 'MNISTUSPS':
        mat = sio.loadmat(os.path.join(args.data_path, 'MNIST-USPS.mat'))
        data_X.append(mat['X1'].astype('float32'))  # (5000,784)
        data_X.append(mat['X2'].astype('float32'))  # (5000,784)
        label_y = np.squeeze(mat['Y'])

    elif args.dataset == 'YoutubeFace50':
        mat = hdf5storage.loadmat(os.path.join(args.data_path, 'YouTubeFace50_4Views.mat'))
        x1 = mat['X'][0][0]
        x2 = mat['X'][1][0]
        # X_list.append(X[1].astype('float32'))              # (4485,59)
        # X_list.append(X[0].astype('float32'))              # (4485,20)
        # Y_list.append(np.squeeze(mat['Y']))
        # print(np.squeeze(mat['Y']))


        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['Y'])
        index = [i for i in range(126054)]
        np.random.seed(576)
        np.random.shuffle(index)
        for i in range(126054):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['Y'][index[i]]

        # from sklearn.preprocessing import normalize
        # xx1 = normalize(xx1, axis=1, norm='max')
        # xx2 = normalize(xx2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        data_X.append(xx1)
        data_X.append(xx2)
        y = np.squeeze(Y).astype('int')
        label_y = y

    elif args.dataset == 'aloideep3v':
        mat = sio.loadmat(os.path.join(args.data_path, 'aloideep3v.mat'))
        X = mat['X'][0]
        # X_list.append(X[1].astype('float32'))              # (4485,59)
        # X_list.append(X[0].astype('float32'))              # (4485,20)
        # Y_list.append(np.squeeze(mat['Y']))
        # print(np.squeeze(mat['truth']))

        x1 = X[1]
        x2 = X[2]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['truth'])
        index = [i for i in range(10800)]
        np.random.seed(10800)
        np.random.shuffle(index)
        for i in range(10800):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['truth'][index[i]]

        # from sklearn.preprocessing import normalize
        # xx1 = normalize(xx1, axis=1, norm='max')
        # xx2 = normalize(xx2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        data_X.append(xx1)
        data_X.append(xx2)
        y = np.squeeze(Y).astype('int')
        label_y = y

    elif args.dataset =='NoisyMNIST':
        Y_list = []
        mat = sio.loadmat(os.path.join(args.data_path,'NoisyMNIST.mat'))
        train = DataSet_NoisyMNIST(mat['X1'], mat['X2'], mat['trainLabel'])
        tune = DataSet_NoisyMNIST(mat['XV1'], mat['XV2'], mat['tuneLabel'])
        test = DataSet_NoisyMNIST(mat['XTe1'], mat['XTe2'], mat['testLabel'])
        # X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
        # X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
        # Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        data_X.append(np.concatenate([train.images1, tune.images1, test.images1], axis=0))
        data_X.append(np.concatenate([train.images2, tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate([np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        # print(Y_list[0])
        x1 = data_X[0]
        x2 = data_X[1]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(Y_list[0])
        index = [i for i in range(70000)]
        np.random.seed(784)
        np.random.shuffle(index)
        for i in range(70000):
            xx1[i] = x1[index[i]]                    # (70000, 784)
            xx2[i] = x2[index[i]]                    # (70000, 784)
            Y[i] = Y_list[0][index[i]]
        # print(Y)
        data_X = [xx1, xx2]
        label_y = Y


    else:
        raise 'Unknown Dataset'

    if args.data_norm == 'standard':
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == 'l2-norm':
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == 'min-max':
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    args.n_sample = data_X[0].shape[0]
    return data_X, label_y


def load_dataset(args):
    data, targets = load_mat(args)
    dataset = IncompleteMultiviewDataset(args.n_views, data, targets, args.missing_rate)
    return dataset



class DataSet_NoisyMNIST(object):
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]

class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.targets = label_y - np.min(label_y)

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return idx, data, label


import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


class IncompleteMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y, missing_rate):
        super(IncompleteMultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.targets = label_y - np.min(label_y)

        self.missing_mask = torch.from_numpy(self._get_mask(n_views, self.data[0].shape[0], missing_rate)).bool()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        mask = self.missing_mask[idx]
        return idx, data, mask, label

    @staticmethod
    def _get_mask(view_num, alldata_len, missing_rate):
        """Randomly generate incomplete data information, simulate partial view data with complete view data
        :param view_num:view number
        :param alldata_len:number of samples
        :param missing_rate:Defined in section 4.1 of the paper
        :return: mask
        """
        full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))

        alldata_len = alldata_len - int(alldata_len * (1 - missing_rate))
        missing_rate = 0.5
        if alldata_len != 0:
            one_rate = 1.0 - missing_rate
            if one_rate <= (1 / view_num):
                enc = OneHotEncoder()  # n_values=view_num
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            error = 1
            if one_rate == 1:
                matrix = randint(1, 2, size=(alldata_len, view_num))
                full_matrix = np.concatenate([matrix, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            while error >= 0.005:
                enc = OneHotEncoder()  # n_values=view_num
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                one_num = view_num * alldata_len * one_rate - alldata_len
                ratio = one_num / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
                one_num_iter = one_num / (1 - a / one_num)
                ratio = one_num_iter / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
                ratio = np.sum(matrix) / (view_num * alldata_len)
                error = abs(one_rate - ratio)
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)

        choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
        matrix = full_matrix[choice]
        return matrix


class IncompleteDatasetSampler:
    def __init__(self, dataset: Dataset, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.compelte_idx = torch.where(self.dataset.missing_mask.sum(dim=1) == self.dataset.n_views)[0]
        self.num_samples = self.compelte_idx.shape[0]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(self.num_samples, generator=g).tolist()

        indices = self.compelte_idx[indices].tolist()

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DatasetWithIndex(Dataset):
    def __getitem__(self, idx):
        img, label = super(DatasetWithIndex, self).__getitem__(idx)
        return idx, img, label
