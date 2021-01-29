#  python -m egg.zoo.visual_ref.pre_train --vocab_size=10 --n_epochs=10 --random_seed=7 --lr=1e-3 --batch_size=32 --optimizer=adam

from __future__ import print_function

import math
import pickle
from pathlib import Path
import os

from sklearn.model_selection import train_test_split

import numpy as np

import torch
import torch.distributions
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

import egg.core as core
from egg.zoo.visual_ref.dataset import CaptionDataset, pad_collate
from egg.zoo.visual_ref.models import Vision, ImageCaptioner
from egg.zoo.visual_ref.preprocess import IMAGES_FILENAME, CAPTIONS_FILENAME, VOCAB_FILENAME, MAX_CAPTION_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 1

CHECKPOINT_PATH_VISION = os.path.join(Path.home(), "data/egg/visual_ref/checkpoints/vision.pt")
CHECKPOINT_PATH_IMAGE_CAPTIONING = os.path.join(Path.home(), "data/egg/visual_ref/checkpoints/image_captioning.pt")

DATA_PATH = os.path.expanduser("~/data/abstract_scenes/preprocessed/")
DATASET_SIZE = 60396

NUM_EPOCHS = 10

VAL_INTERVAL = 100

PRINT_SAMPLE_CAPTIONS = 10

def print_caption(caption, vocab):
    words = [vocab.itos[word] for word in caption]
    print(" ".join(words))

def print_model_output(output, target_captions, vocab, num_captions=1):
    captions_model = torch.argmax(output, dim=1)
    for i in range(num_captions):
        print("Target: ", end="")
        print_caption(target_captions[i], vocab)
        print("Model output: ", end="")
        print_caption(captions_model[i], vocab)

def print_sample_model_output(model, dataloader, vocab, num_captions=1):
    images, captions, caption_lengths = next(iter(dataloader))

    output, decode_lengths = model.forward_test(images)

    print_model_output(output, captions, vocab, num_captions)



def main(params):
    # initialize the egg lib
    opts = core.init(params=params)

    batch_size = opts.batch_size  # set via the CL arguments above

    # create model checkpoint directory
    if not os.path.exists(CHECKPOINT_PATH_IMAGE_CAPTIONING):
        os.makedirs(CHECKPOINT_PATH_IMAGE_CAPTIONING)

    #TODO improve
    all_indices = list(range(DATASET_SIZE))

    train_indices, test_indices = train_test_split(all_indices, test_size=0.1, random_state=RANDOM_SEED)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=RANDOM_SEED)

    train_loader = DataLoader(
        CaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME,
            CAPTIONS_FILENAME,
            train_indices,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate,
    )
    val_images_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            DATA_PATH,
            IMAGES_FILENAME,
            CAPTIONS_FILENAME,
            val_indices,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=pad_collate,
    )

    vocab_path = os.path.join(DATA_PATH, VOCAB_FILENAME)
    print("Loading vocab from {}".format(vocab_path))
    with open(vocab_path, "rb") as file:
        vocab = pickle.load(file)

    # TODO: embedding size for speaker is 1024 in paper
    embedding_size = 512
    lstm_hidden_size = 512
    model_visual_encoder = Vision(embedding_size, fine_tune_resnet=False)
    model_image_captioning = ImageCaptioner(model_visual_encoder, embedding_size, lstm_hidden_size, vocab, MAX_CAPTION_LEN)

    # uses command-line parameters we passed to core.init
    optimizer = core.build_optimizer(model_image_captioning.parameters())

    model_image_captioning = model_image_captioning.to(device)

    def save_model(model, optimizer, best_val_loss, epoch):
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model_visual_encoder.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': mean_loss,
        # }, CHECKPOINT_PATH_VISION)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, CHECKPOINT_PATH_IMAGE_CAPTIONING)


    def validate_model(model, dataloader):
        print(f"EVAL")
        model.eval()

        print_sample_model_output(model, dataloader, vocab, PRINT_SAMPLE_CAPTIONS)

        val_losses = []
        for batch_idx, (images, captions, caption_lengths) in enumerate(dataloader):
            output, decode_lengths = model.forward_test(images)

            loss = model.calc_loss(output, captions, caption_lengths)

            val_losses.append(loss.mean().item())

        val_loss = np.mean(val_losses)
        print(f"val loss: {val_loss}")

        model.train()
        return val_loss

    best_val_loss = math.inf
    for epoch in range(NUM_EPOCHS):
        losses = []
        for batch_idx, (images, captions, caption_lengths) in enumerate(train_loader):
            output = model_image_captioning(images, captions, caption_lengths)

            loss = model_image_captioning.calc_loss(output, captions, caption_lengths)
            losses.append(loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % VAL_INTERVAL == 0:
                print(f"Batch {batch_idx}: train loss: {np.mean(losses)}")
                val_loss = validate_model(model_image_captioning, val_images_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model_image_captioning, optimizer, best_val_loss, epoch)

        print(f'Train Epoch: {epoch}, train loss: {np.mean(losses)}')
        val_loss = validate_model(model_image_captioning, val_images_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model_image_captioning, optimizer, best_val_loss, epoch)


    core.close()





if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

