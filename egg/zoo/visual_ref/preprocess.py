"""Preprocess the abstract scenes images and captions and store them in a hdf5 file"""

import argparse
import os
import pickle
import string
import sys

from collections import Counter
import imageio

import matplotlib.pyplot as plt

import h5py
import nltk
from torchtext.vocab import Vocab
from tqdm import tqdm

nltk.download("punkt")

VOCAB_FILENAME = "vocab.p"
IMAGES_FILENAME = "images.hdf5"
CAPTIONS_FILENAME = "captions.p"


def encode_caption(caption, vocab):
    return (
        [vocab.stoi[TOKEN_START]]
        + [vocab.stoi[word] for word in caption]
        + [vocab.stoi[TOKEN_END]]
    )

MAX_CAPTION_LEN = 25
TOKEN_PADDING = "<pad>"
TOKEN_START = "<sos>"
TOKEN_END = "<eos>"

def show_image(img_data):
    plt.imshow(img_data), plt.axis('off')
    plt.show()

def preprocess_images_and_captions(
    dataset_folder,
    output_folder,
    vocabulary_size,
):
    images = []
    captions = []
    word_freq = Counter()

    images_folder = os.path.join(dataset_folder, "RenderedScenes")
    for img_filename in tqdm(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, img_filename)
        img = imageio.imread(img_path)

        # discard transparency channel
        img = img[..., :3]

        # show_image(img)
        images.append(img)

    captions_file_1 = os.path.join(dataset_folder, "SimpleSentences", "SimpleSentences1_10020.txt")
    captions_file_2 = os.path.join(dataset_folder, "SimpleSentences", "SimpleSentences2_10020.txt")

    for captions_file in [captions_file_1, captions_file_2]:
        with open(captions_file) as file:
            for line in file:
                splitted = line.split("\t")
                if len(splitted) > 1:
                    # print(line)
                    image_id = splitted[0]
                    caption_id = splitted[1]
                    caption = splitted[2]

                    # remove special chars, make caption lower case
                    caption = caption.replace("\n", "").replace('"', "").lower()
                    caption = caption.translate(
                        str.maketrans(dict.fromkeys(string.punctuation))
                    )

                    # Tokenize the caption
                    caption = nltk.word_tokenize(caption)

                    # Cut off too long captions
                    caption = caption[:MAX_CAPTION_LEN]

                    word_freq.update(caption)

                    captions.append({
                        "image_id": image_id,
                        "caption_id": caption_id,
                        "caption": caption,
                    })

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Print the most frequent words
    print(f"Most frequent words: {word_freq.most_common(10)}")

    # Create vocab
    vocab = Vocab(word_freq, specials=[TOKEN_PADDING, Vocab.UNK, TOKEN_START, TOKEN_END], max_size=vocabulary_size)
    vocab_path = os.path.join(output_folder, VOCAB_FILENAME)

    print("Saving new vocab to {}".format(vocab_path))
    with open(vocab_path, "wb") as file:
        pickle.dump(vocab, file)

    # Create hdf5 file and dataset for the images
    images_dataset_path = os.path.join(output_folder, IMAGES_FILENAME)
    print("Creating image dataset at {}".format(images_dataset_path))
    with h5py.File(images_dataset_path, "a") as h5py_file:
        for img_id, img in tqdm(enumerate(images)):
            # Read image and save it to hdf5 file
            h5py_file.create_dataset(
                str(img_id), (3, 400, 500), dtype="uint8", data=img
            )

    # Encode captions
    for caption in captions:
        caption["caption"] = encode_caption(caption["caption"], vocab)

    # Save captions
    captions_path = os.path.join(output_folder, CAPTIONS_FILENAME)
    print("Saving captions to {}".format(captions_path))
    with open(captions_path, "wb") as file:
        pickle.dump(captions, file)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the abstract scenes dataset is located",
        default=os.path.expanduser("~/data/abstract_scenes/AbstractScenes_v1.1/"),
    )
    parser.add_argument(
        "--output-folder",
        help="Folder in which the preprocessed data should be stored",
        default=os.path.expanduser("~/data/abstract_scenes/preprocessed/"),
    )
    parser.add_argument(
        "--vocabulary-size",
        help="Number of words that should be saved in the vocabulary",
        type=int,
        default=2685,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    preprocess_images_and_captions(
        parsed_args.dataset_folder,
        parsed_args.output_folder,
        parsed_args.vocabulary_size,
    )
