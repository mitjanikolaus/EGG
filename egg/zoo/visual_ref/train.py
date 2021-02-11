#  python -m egg.zoo.visual_ref.train --vocab_size=10 --n_epochs=15 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam
import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from torch.nn import functional as F

import matplotlib.pyplot as plt

import egg.core as core
from egg.core import ConsoleLogger, Callback, Interaction
from egg.zoo.visual_ref.dataset import VisualRefCaptionDataset
from egg.zoo.visual_ref.game import OracleSenderReceiverRnnSupervised
from egg.zoo.visual_ref.models import (
    VisualRefSpeakerDiscriminativeOracle,
    VisualRefListenerOracle,
    ImageSentenceRanker,
)
from egg.zoo.visual_ref.preprocess import (
    DATA_PATH,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    VOCAB_FILENAME,
)
from egg.zoo.visual_ref.train_image_sentence_ranking import (
    CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING,
)
from egg.zoo.visual_ref.utils import decode_caption, VisualRefLoggingStrategy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrintDebugEvents(Callback):
    def __init__(self, train_dataset, args):
        super().__init__()

        vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
        with open(vocab_path, "rb") as file:
            self.vocab = pickle.load(file)

        self.train_loss = 0
        self.train_accuracies = 0
        self.args = args

        self.train_dataset = train_dataset

    def print_sample_interactions(self, interaction_logs, num_interactions=5):
        target_image_ids, distractor_image_ids = interaction_logs.sender_input
        for z in range(num_interactions):
            target_image = self.train_dataset.get_image_features(
                int(target_image_ids[z]), channels_first=False, normalize=False
            )
            distractor_image = self.train_dataset.get_image_features(
                int(distractor_image_ids[z]), channels_first=False, normalize=False
            )

            # plot the two images side-by-side
            image = torch.cat([target_image, distractor_image], dim=1).cpu().numpy()

            target_position = interaction_logs.labels[z]
            receiver_guess = torch.argmax(interaction_logs.receiver_output[z])

            message = decode_caption(interaction_logs.message[z], self.vocab)
            plt.title(
                f"Left: Target, Right: Distractor"
                f"\nReceiver guess correct: {target_position == receiver_guess}"
                f"\nMessage: {message}"
            )
            plt.imshow(image)
            plt.show()

    def on_test_end(self, _loss, interaction_logs: Interaction, epoch: int):
        pass

    def on_batch_end(
        self,
        interaction_logs: Interaction,
        loss: float,
        batch_id: int,
        is_training: bool = True,
    ):
        if batch_id == 0:
            self.train_loss = 0
            self.train_accuracies = 0

        self.train_loss += loss.detach()
        self.train_accuracies += interaction_logs.aux["acc"].sum()

        if (batch_id + 1) % self.args.log_frequency == 0:
            mean_loss = self.train_loss / self.args.log_frequency
            batch_size = interaction_logs.aux["acc"].size()[0]
            mean_acc = self.train_accuracies / (self.args.log_frequency * batch_size)

            print(
                f"Batch {batch_id + 1}: loss: {mean_loss:.3f} accuracy: {mean_acc:.3f}"
            )

            self.train_loss = 0
            self.train_accuracies = 0

            if self.args.debug:
                self.print_sample_interactions(interaction_logs)


def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
    # in the discriminative case, accuracy is computed by comparing the index with highest score in Receiver output (a distribution of unnormalized
    # probabilities over target poisitions) and the corresponding label read from input, indicating the ground-truth position of the target
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    # similarly, the loss computes cross-entropy between the Receiver-produced target-position probability distribution and the labels
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def main(args):
    train_dataset = VisualRefCaptionDataset(
        DATA_PATH, IMAGES_FILENAME["train"], CAPTIONS_FILENAME["train"], args.batch_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        VisualRefCaptionDataset(
            DATA_PATH, IMAGES_FILENAME["val"], CAPTIONS_FILENAME["val"], args.batch_size
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # TODO
    # TODO: embedding size for speaker is 1024 in paper
    args.sender_hidden = 1024  # TODO
    args.sender_embedding = 512  # ???
    args.receiver_embedding = 100  # ???
    args.receiver_hidden = 512  # ???
    args.sender_entropy_coeff = 0.0  # entropy regularization
    args.receiver_entropy_coeff = 0.0  # entropy regularization
    args.sender_cell = "lstm"
    args.receiver_cell = "lstm"
    args.vocab_size = len(vocab)
    args.max_len = 25
    args.random_seed = 1

    word_embedding_size = 100
    joint_embeddings_size = 512
    lstm_hidden_size = 512
    checkpoint_ranking_model = torch.load(CHECKPOINT_PATH_IMAGE_SENTENCE_RANKING, map_location=device)
    ranking_model = ImageSentenceRanker(
        word_embedding_size,
        joint_embeddings_size,
        lstm_hidden_size,
        len(vocab),
        fine_tune_resnet=False,
    )
    ranking_model.load_state_dict(checkpoint_ranking_model["model_state_dict"])

    sender = VisualRefSpeakerDiscriminativeOracle(DATA_PATH, CAPTIONS_FILENAME, args.max_len)
    receiver = VisualRefListenerOracle(ranking_model)

    # sender = core.RnnSenderReinforce(sender, vocab_size=opts.vocab_size, embed_dim=opts.sender_embedding,
    #                                  hidden_size=opts.sender_hidden, cell=opts.sender_cell, max_len=opts.max_len)
    # receiver = core.RnnReceiverDeterministic(receiver, vocab_size=opts.vocab_size, embed_dim=opts.receiver_embedding,
    #                                          hidden_size=opts.receiver_hidden, cell=opts.receiver_cell)

    # use LoggingStrategy that stores image IDs
    # train_logging_strategy = LoggingStrategy(store_sender_input=False, store_receiver_input=False)
    train_logging_strategy = VisualRefLoggingStrategy()
    game = OracleSenderReceiverRnnSupervised(
        sender,
        receiver,
        loss,
        receiver_entropy_coeff=args.receiver_entropy_coeff,
        train_logging_strategy=train_logging_strategy,
    )

    callbacks = [ConsoleLogger(print_train_loss=True, as_json=False)]
    # core.PrintValidationEvents(n_epochs=1)
    callbacks.append(PrintDebugEvents(train_dataset, args))

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
    )

    print("Starting training with args: ")
    print(args)
    print("Number of samples: ", len(train_dataset))
    trainer.train(args.n_epochs)

    game.eval()

    core.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Print debug interactions output.",
    )
    parser.add_argument(
        "--log-frequency",
        default=100,
        type=int,
        help="Logging frequency (number of batches)",
    )

    # initialize the egg lib
    # get pre-defined common line arguments (batch/vocab size, etc).
    # See egg/core/util.py for a list
    args = core.init(parser)

    return args


if __name__ == "__main__":
    print("Start training on device: ", device)
    args = get_args()
    main(args)
