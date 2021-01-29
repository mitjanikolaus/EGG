#  python -m egg.zoo.visual_ref.train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam

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
from egg.zoo.visual_ref.models import VisualRefDiscriReceiver, VisualRefSenderFunctional, \
    VisualRefSpeakerDiscriminativeOracle, VisualRefListenerOracle
from egg.zoo.visual_ref.pre_train import Vision
from egg.zoo.visual_ref.preprocess import DATA_PATH, IMAGES_FILENAME, CAPTIONS_FILENAME, DATASET_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrintDebugEvents(Callback):
    def __init__(self):
        super().__init__()

    def on_test_end(self, _loss, interaction_logs: Interaction, epoch: int):
        for z in range(10):
            input_0 = interaction_logs.sender_input[z].squeeze(1)[0]
            input_1 = interaction_logs.sender_input[z].squeeze(1)[1]

            # plot the two images side-by-side
            image = torch.cat([input_0, input_1], dim=1).cpu().numpy()

            target_position = interaction_logs.labels[z]
            receiver_guess = torch.argmax(interaction_logs.receiver_output[z])

            plt.title(f"Target position: {target_position}, channel message {interaction_logs.message[z]}, "
                      f"Receiver guess: {receiver_guess}")
            plt.imshow(image, cmap='gray')
            plt.show()



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

    # kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    # mnist_dataset_train = datasets.MNIST('./data', train=True, download=True,
    #                transform=transforms.ToTensor())
    # n_samples = 10000
    # dataset_train = VisualRefCaptionDataset(mnist_dataset_train, n_samples)
    # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, **kwargs)

    if opts.batch_size > 1:
        raise NotImplementedError("Batch size greater than 1 not supported yet.")

    # TODO improve splits
    all_indices = list(range(DATASET_SIZE))

    train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=RANDOM_SEED)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=RANDOM_SEED)

    train_loader = DataLoader(
        VisualRefCaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME,
            CAPTIONS_FILENAME,
            train_indices,
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        VisualRefCaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME,
            CAPTIONS_FILENAME,
            val_indices,
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # mnist_dataset_test = datasets.MNIST('./data', train=False, download=True,
    #                                      transform=transforms.ToTensor())
    # n_samples = 1000
    # dataset_test = VisualRefDiscriDataset(mnist_dataset_test, n_samples)
    # test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=opts.batch_size, shuffle=True, **kwargs)

    # TODO
    opts.sender_hidden = 1024 # TODO
    opts.sender_embedding = 512 #???
    opts.receiver_embedding = 100 #???
    opts.receiver_hidden = 512 #???
    opts.sender_entropy_coeff = 0.0 # entropy regularization
    opts.sender_cell = "lstm"
    opts.receiver_cell = "lstm"
    opts.vocab_size = 10
    opts.max_len = 1
    opts.random_seed = 1

    # TODO
    n_features = opts.sender_embedding

    # checkpoint_vision = torch.load(CHECKPOINT_PATH)
    # vision = Vision(n_features)
    # vision.load_state_dict(checkpoint_vision['model_state_dict'])

    sender = VisualRefSpeakerDiscriminativeOracle(n_hidden=opts.sender_embedding, n_features=n_features)
    # receiver = VisualRefListenerOracle(vision=vision, n_features=n_features, n_hidden=opts.receiver_hidden)

    sender = core.RnnSenderReinforce(sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_embedding,
                                     hidden_size=opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len)
    # receiver = core.RnnReceiverDeterministic(receiver, vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding,
    #                                          hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)

    # game = core.SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=opts.sender_entropy_coeff,
    #                                        receiver_entropy_coeff=0)

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


    # test_inputs = []
    # for z in range(10):
    #     index = (test_loader.dataset.targets[:100] == z).nonzero()[0, 0]
    #     img, _ = test_loader.dataset[index]
    #     test_inputs.append(img.unsqueeze(0))
    # test_inputs = torch.cat(test_inputs)
    #
    # test_dataset = [[test_inputs, None]]
    #
    # def plot(game, test_dataset, is_gs, variable_length):
    #     interaction = \
    #         core.dump_interactions(game, test_dataset, is_gs, variable_length)
    #
    #     for z in range(10):
    #         src = interaction.sender_input[z].squeeze(0)
    #         dst = interaction.receiver_output[z].view(28, 28)
    #         # we'll plot two images side-by-side: the original (left) and the reconstruction
    #         image = torch.cat([src, dst], dim=1).cpu().numpy()
    #
    #         plt.title(f"Input: digit {z}, channel message {interaction.message[z]}")
    #         plt.imshow(image, cmap='gray')
    #         plt.show()

    # plot(game_rnn, test_dataset, is_gs=False, variable_length=True)
    #
    # f, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    #
    # for x in range(10):
    #     for y in range(10):
    #
    #         t = torch.zeros((1, 2)).to(device).long()
    #         t[0, 0] = x
    #         t[0, 1] = y
    #
    #         with torch.no_grad():
    #             sample = game_rnn.receiver(t)[0].float().cpu()
    #             sample = sample[0, :].view(28, 28)
    #             ax[x][y].imshow(sample, cmap='gray')
    #
    #             if y == 0:
    #                 ax[x][y].set_ylabel(f'x={x}')
    #             if x == 0:
    #                 ax[x][y].set_title(f'y={y}')
    #
    #             ax[x][y].set_yticklabels([])
    #             ax[x][y].set_xticklabels([])
    # plt.show()

    core.close()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

