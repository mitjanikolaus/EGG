import os
import pickle

import h5py as h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from egg.zoo.visual_ref.preprocess import DATA_PATH, VOCAB_FILENAME
from egg.zoo.visual_ref.utils import print_caption, show_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These input-data-processing classes take input data from a text file and convert them to the format
# appropriate for the recognition and discrimination games, so that they can be read by
# the standard pytorch DataLoader. The latter requires the data reading classes to support
# a __len__(self) method, returning the size of the dataset, and a __getitem__(self, idx)
# method, returning the idx-th item in the dataset. We also provide a get_n_features(self) method,
# returning the dimensionality of the Sender input vector after it is transformed to one-hot format.

# The AttValDiscriDataset class, used in the discrimination game takes an input file with a variable
# number of period-delimited fields, where all fields but the last represent attribute-value vectors
# (with space-delimited attributes). The last field contains the index (counting from 0) of the target
# vector.
# Here, we create a data-frame containing 3 fields: sender_input, labels and receiver_input (these are
# expected by EGG, the first two mandatorily so).
# The sender_input corresponds to the target vector (in one-hot format), labels are the indices of the
# target vector location and receiver_input is a matrix with a row for each input vector (in input order).


class VisualRefDiscriDataset(Dataset):
    def __init__(self, mnist_dataset, n_samples):
        # construct dataset
        self.data = []

        sample_ids_target = np.random.choice(range(len(mnist_dataset)), size=n_samples, replace=False)
        sample_ids_distractor = np.random.choice(range(len(mnist_dataset)), size=n_samples, replace=False)
        for id_target, id_distractor in zip(sample_ids_target, sample_ids_distractor):
            target_img, _ = mnist_dataset.__getitem__(id_target)
            distractor_img, _ = mnist_dataset.__getitem__(id_distractor)

            # The sender always gets the target first
            sender_input = torch.stack([target_img, distractor_img])

            # The receiver get target and distractor in random order
            target_position = np.random.choice(2)
            if target_position == 0:
                receiver_input = torch.stack([target_img, distractor_img])
            else:
                receiver_input = torch.stack([distractor_img, target_img])
            target_label = target_position

            self.data.append((sender_input, target_label, receiver_input))


        #
        #
        # frame = open(path,'r')
        # self.frame = []
        # for row in frame:
        #     raw_info = row.split('.')
        #     index_vectors = list([list(map(int,x.split())) for x in raw_info[:-1]])
        #     target_index = int(raw_info[-1])
        #     target_one_hot = []
        #     for index in index_vectors[target_index]:
        #         current=np.zeros(n_values)
        #         current[index]=1
        #         target_one_hot=np.concatenate((target_one_hot,current))
        #     target_one_hot_tensor = torch.FloatTensor(target_one_hot)
        #     one_hot = []
        #     for index_vector in index_vectors:
        #         for index in index_vector:
        #             current=np.zeros(n_values)
        #             current[index]=1
        #             one_hot=np.concatenate((one_hot,current))
        #     one_hot_sequence = torch.FloatTensor(one_hot).view(len(index_vectors),-1)
        #     label= torch.tensor(target_index)
        #     self.frame.append((target_one_hot_tensor,label,one_hot_sequence))
        # frame.close()

    def get_n_features(self):
        # TODO
        raise NotImplementedError()
        return self.data[0][0].size(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    CAPTIONS_PER_IMAGE = 6

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        normalize=None,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(
            os.path.join(data_folder, features_filename), "r"
        )

        self.features_scale_factor = features_scale_factor

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        # Set pytorch transformation pipeline
        self.transform = normalize

    def get_image_features(self, id):
        image_data = self.images[str(id)][()]

        # scale the features with given factor
        image_data = image_data * self.features_scale_factor

        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, i):
        image_id = i // self.CAPTIONS_PER_IMAGE
        caption_id = i % self.CAPTIONS_PER_IMAGE

        image = self.get_image_features(image_id)

        caption = self.captions[image_id][caption_id]
        caption = torch.LongTensor(
            caption
        )

        return image, caption

    def __len__(self):
        return len(self.images) * self.CAPTIONS_PER_IMAGE

    def pad_collate(batch):
        images = torch.stack([s[0] for s in batch])
        captions = [s[1] for s in batch]

        sequence_lengths = torch.tensor([len(c) for c in captions])
        padded_captions = pad_sequence(captions, batch_first=True)

        return images.to(device), padded_captions.to(device), sequence_lengths.to(device)


class VisualRefCaptionDataset(Dataset):
    """
    PyTorch Dataset that provides sets of target and distractor images and captions
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        normalize=None,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param data_indices: dataset split, indices of images that should be included
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(
            os.path.join(data_folder, features_filename), "r"
        )

        self.features_scale_factor = features_scale_factor

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        # Set pytorch transformation pipeline
        self.transform = normalize

        self.sample_image_ids = []
        for i in range(len(self.images)):
            for j in range(len(self.images)):
                self.sample_image_ids.append((i, j))


    def get_image_features(self, id):
        image_data = self.images[str(id)][()]

        # scale the features with given factor
        image_data = image_data * self.features_scale_factor

        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, i):
        target_image_id, distractor_image_id = self.sample_image_ids[i]

        target_image = self.get_image_features(target_image_id)
        distractor_image = self.get_image_features(distractor_image_id)

        # target_captions = self.captions[target_image_id]
        # distractor_captions = self.captions[distractor_image_id]

        # The sender always gets the target first
        sender_input = target_image, distractor_image, target_image_id, distractor_image_id #, target_captions, distractor_captions
        #torch.stack([target_image, distractor_image]),

        # The receiver gets target and distractor in random order
        target_position = np.random.choice(2)
        if target_position == 0:
            receiver_input = target_image, distractor_image
        else:
            receiver_input = distractor_image, target_image
        target_label = target_position

        return sender_input, target_label, receiver_input


    def __len__(self):
        return len(self.images) ** 2

    # def pad_collate(batch):
    #     sender_inputs = [s[0] for s in batch]
    #     target_labels = [s[1] for s in batch]
    #     receiver_inputs = [s[2] for s in batch]
    #
    #     sequence_lengths = torch.tensor([len(c) for c in captions])
    #     padded_captions = pad_sequence(captions, batch_first=True)
    #
    #     return images.to(device), padded_captions.to(device), sequence_lengths.to(device)

