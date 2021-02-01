#  python -m egg.zoo.visual_ref.train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam
import os
import pickle

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import egg.core as core

from torchvision import datasets, transforms
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt
import random
import numpy as np

import egg.core as core
from egg.core import ConsoleLogger, Callback, Interaction
from egg.zoo.visual_ref.dataset import VisualRefCaptionDataset
from egg.zoo.visual_ref.game import OracleSenderReceiverRnnReinforce
from egg.zoo.visual_ref.models import VisualRefDiscriReceiver, VisualRefSenderFunctional, \
    VisualRefSpeakerDiscriminativeOracle, VisualRefListenerOracle
from egg.zoo.visual_ref.train_image_captioning import Vision
from egg.zoo.visual_ref.preprocess import DATA_PATH, IMAGES_FILENAME, CAPTIONS_FILENAME, DATASET_SIZE, RANDOM_SEED, \
    VOCAB_FILENAME
from egg.zoo.visual_ref.utils import decode_caption

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_INTERVAL = 10

class PrintDebugEvents(Callback):
    def __init__(self):
        super().__init__()

        vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
        with open(vocab_path, "rb") as file:
            self.vocab = pickle.load(file)

    def print_sample_interactions(self, interaction_logs, num_interactions=5):
        for z in range(num_interactions):
            image_0 = interaction_logs.receiver_input[0][z]
            image_1 = interaction_logs.receiver_input[1][z]

            image_0 = image_0.reshape(image_0.shape[1], image_0.shape[2], image_0.shape[0])
            image_1 = image_1.reshape(image_1.shape[1], image_1.shape[2], image_1.shape[0])

            # plot the two images side-by-side
            image = torch.cat([image_0, image_1], dim=1).cpu().numpy()

            target_position = interaction_logs.labels[z]
            receiver_guess = torch.argmax(interaction_logs.receiver_output[z])

            message = decode_caption(interaction_logs.message[z], self.vocab)
            plt.title(f"Target position: {target_position}, Receiver guess: {receiver_guess}"
                      f"\nMessage: {message}")
            plt.imshow(image)
            plt.show()

    def on_test_end(self, _loss, interaction_logs: Interaction, epoch: int):
        self.print_sample_interactions(interaction_logs)

    def on_batch_end(
            self, interaction_logs: Interaction, loss: float, batch_id: int, is_training: bool = True
    ):
        if batch_id % LOG_INTERVAL == 0:
            accuracy = interaction_logs.aux['acc'].mean()
            print(f"Batch {batch_id}: loss: {loss} accuracy: {accuracy}")
            # self.print_sample_interactions(interaction_logs)


def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {'acc': acc}


def main(params):
    # initialize the egg lib
    opts = core.init(params=params)
    # get pre-defined common line arguments (batch/vocab size, etc).
    # See egg/core/util.py for a list

    train_loader = DataLoader(
        VisualRefCaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME["train"],
            CAPTIONS_FILENAME["train"],
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        # collate_fn=VisualRefCaptionDataset.pad_collate
    )
    val_loader = DataLoader(
        VisualRefCaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME["val"],
            CAPTIONS_FILENAME["val"],
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        # collate_fn=VisualRefCaptionDataset.pad_collate
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # TODO
    opts.sender_hidden = 1024 # TODO
    opts.sender_embedding = 512 #???
    opts.receiver_embedding = 100 #???
    opts.receiver_hidden = 512 #???
    opts.sender_entropy_coeff = 0.0 # entropy regularization
    opts.sender_cell = "lstm"
    opts.receiver_cell = "lstm"
    opts.vocab_size = len(vocab)
    opts.max_len = 1
    opts.random_seed = 1

    # TODO
    n_features = opts.sender_embedding

    # checkpoint_vision = torch.load(CHECKPOINT_PATH)
    # vision = Vision(n_features)
    # vision.load_state_dict(checkpoint_vision['model_state_dict'])

    sender = VisualRefSpeakerDiscriminativeOracle(DATA_PATH, CAPTIONS_FILENAME)
    receiver = VisualRefListenerOracle(n_features=n_features, n_hidden=opts.receiver_hidden)

    # sender = core.RnnSenderReinforce(sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_embedding,
    #                                  hidden_size=opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len)
    receiver = core.RnnReceiverDeterministic(receiver, vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding,
                                             hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)

    game = OracleSenderReceiverRnnReinforce(sender, receiver, loss, receiver_entropy_coeff=0)

    callbacks = [ConsoleLogger(print_train_loss=True, as_json=False)]
    # core.PrintValidationEvents(n_epochs=1)
    callbacks.append(PrintDebugEvents())

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_loader, validation_data=val_loader,
                           callbacks=callbacks)

    print("Starting training with opts: ")
    print(opts)
    trainer.train(opts.n_epochs)

    game.eval()

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

