import os
import pickle

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchvision.models import resnet50

from egg.zoo.visual_ref.preprocess import TOKEN_START, TOKEN_END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vision(nn.Module):
    def __init__(self, embedding_size, fine_tune_resnet=True):
        super(Vision, self).__init__()
        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, embedding_size)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class ImageCaptioner(nn.Module):
    def __init__(self, visual_encoder, word_embedding_size, visual_embedding_size, lstm_hidden_size, vocab, max_caption_length):
        super(ImageCaptioner, self).__init__()
        self.visual_encoder = visual_encoder
        self.lstm_hidden_size = lstm_hidden_size

        self.vocab = vocab
        self.vocab_size = len(vocab)

        #TODO no word embeddings used in paper?
        self.word_embedding = nn.Embedding(self.vocab_size, word_embedding_size)

        self.lstm = nn.LSTM(input_size=word_embedding_size + visual_embedding_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, self.vocab_size)

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.max_caption_length = max_caption_length

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_hidden_size).to(device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(device))

    def forward(self, images, captions, caption_lengths):
        image_features = self.visual_encoder(images)
        batch_size = images.shape[0]

        hidden = self.init_hidden(batch_size)

        # TODO should take into account caption length
        # cut off <eos> token as it's not needed to predict with this as input
        captions = captions[:, :-1]

        embedded_captions = self.word_embedding(captions)
        # packed_inputs = pack_padded_sequence(embedded_captions, sequence_lengths, enforce_sorted=False)

        image_features = image_features.unsqueeze(dim=1).repeat(1, embedded_captions.shape[1], 1)

        inputs = torch.cat((image_features, embedded_captions), dim=2)
        lstm_out, hidden = self.lstm(inputs, hidden)
        # output, _ = pad_packed_sequence(lstm_out)

        output = self.fc(lstm_out)
        output = output.transpose(1, 2)
        return output


    def forward_greedy_decode(self, images):
        """ Forward propagation at test time (no teacher forcing)."""
        image_features = self.visual_encoder(images)
        batch_size = images.shape[0]

        decode_lengths = torch.full(
            (batch_size,),
            self.max_caption_length,
            dtype=torch.int64,
            device=device,
        )

        # Initialize LSTM hidden state
        hidden = self.init_hidden(batch_size)

        # Tensors to hold word prediction scores and alphas
        scores = torch.zeros(
            (batch_size, max(decode_lengths), self.vocab_size), device=device
        )

        # At the start, all 'previous words' are the <start> token
        prev_words = torch.full(
            (batch_size,), self.vocab.stoi[TOKEN_START], dtype=torch.int64, device=device
        )

        for t in range(max(decode_lengths)):
            # Find all sequences where an <end> token has been produced in the last timestep
            ind_end_token = (
                torch.nonzero(prev_words == self.vocab.stoi[TOKEN_END])
                .view(-1)
                .tolist()
            )

            # Update the decode lengths accordingly
            decode_lengths[ind_end_token] = torch.min(
                decode_lengths[ind_end_token],
                torch.full_like(decode_lengths[ind_end_token], t, device=device),
            )

            # Check if all sequences are finished:
            indices_incomplete_sequences = torch.nonzero(decode_lengths > t).view(-1)
            if len(indices_incomplete_sequences) == 0:
                break

            # Embed input words
            prev_words_embedded = self.word_embedding(prev_words)

            # Concatenate input: image features and word embeddings
            inputs = torch.cat((image_features, prev_words_embedded), dim=1)

            # Unsqueeze time dimension (1 timestep)
            inputs = inputs.unsqueeze(1)

            # LSTM forward pass
            lstm_out, hidden = self.lstm(inputs, hidden)
            scores_for_timestep = self.fc(lstm_out)
            scores_for_timestep = scores_for_timestep.squeeze(1)

            # Update the previously predicted words
            prev_words = torch.argmax(scores_for_timestep, dim=1)

            scores[indices_incomplete_sequences, t, :] = scores_for_timestep[
                indices_incomplete_sequences
            ]

        scores = scores.transpose(1, 2)
        return scores, decode_lengths


    def calc_loss(self, scores, target_captions, caption_lengths):
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        target_captions = target_captions[:, 1:]

        # Trim produced captions' lengths to target lengths for loss calculation
        scores = scores[:, :, :target_captions.shape[1]]

        return self.loss_function(scores, target_captions)



# In EGG, the game designer must implement the core functionality of the Sender and Receiver agents. These are then
# embedded in wrappers that are used to train them to play Gumbel-Softmax- or Reinforce-optimized games. The core
# Sender must take the input and produce a hidden representation that is then used by the wrapper to initialize
# the RNN or other module that will generate the message. The core Receiver expects a hidden representation
# generated by the message-processing wrapper, plus possibly other game-specific input, and it must generate the
# game-specific output.

# The DiscriReceiver class implements the core Receiver agent for the reconstruction game. In this case, besides the
# vector generated by the message-decoding RNN in the wrapper (x in the forward method), the module also gets game-specific
# Receiver input (_input), that is, the matrix containing all input attribute-value vectors. The module maps these vectors to the
# same dimensionality as the RNN output vector, and computes a dot product between the latter and each of the (transformed) input vectors.
# The output dot prodoct list is interpreted as a non-normalized probability distribution over possible positions of the target.
class VisualRefDiscriReceiver(nn.Module):
    def __init__(self, vision, n_features, n_hidden):
        super(VisualRefDiscriReceiver, self).__init__()
        self.vision = vision
        self.fc1 = nn.Linear(n_hidden, n_features)

        self.fc2 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _input):
        # x: receiver RNN hidden output
        targets = _input[:, 0]
        distractors = _input[:, 1]
        with torch.no_grad():
            emb_targets = self.vision(targets)
            emb_distractors = self.vision(distractors)
        emb_targets = self.fc1(emb_targets)
        emb_distractors = self.fc1(emb_distractors)

        stacked = torch.stack([emb_targets, emb_distractors], dim=1)

        # TODO correct like this?
        dots = torch.matmul(stacked, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()

class VisualRefSenderFunctional(nn.Module):
    def __init__(self, vision, n_hidden, n_features):
        super(VisualRefSenderFunctional, self).__init__()
        self.fc = nn.Linear(n_hidden, n_features)
        self.vision = vision

    def forward(self, x):
        # input: (batch_size, 2, 1, 28, 28)
        targets = x[:, 0]
        distractors = x[:, 1]
        with torch.no_grad():
            emb_targets = self.vision(targets)
            emb_distractors = self.vision(distractors)

        emb_targets = self.fc(emb_targets)
        emb_distractors = self.fc(emb_distractors)

        out = torch.hstack([emb_targets, emb_distractors])
        # out: sender RNN init hidden state
        return out


class VisualRefSpeakerDiscriminativeOracle(nn.Module):
    def __init__(self, data_folder, captions_filename):
        super(VisualRefSpeakerDiscriminativeOracle, self).__init__()

        # Load captions
        with open(os.path.join(data_folder, captions_filename["train"]), "rb") as file:
            self.captions_train = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["val"]), "rb") as file:
            self.captions_val = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["test"]), "rb") as file:
            self.captions_test = pickle.load(file)


    def forward(self, input):
        # input: target_image, distractor_image
        images, target_label, target_image_id, distractor_image_id = input

        #TODO: choose best caption
        output_captions = [self.captions_train[int(i)][0] for i in target_image_id]

        # append end of message token
        # TODO: currently overlap with padding
        for caption in output_captions:
            caption.append(0)

        # Transform lists to tensors
        output_captions = [torch.tensor(caption) for caption in output_captions]

        # Pad all captions in batch to equal length
        output_captions = pad_sequence(output_captions, batch_first=True)

        # out: sender RNN init hidden state
        # out: sequence, logits, entropy
        return output_captions, None, None

class VisualRefListenerOracle(nn.Module):
    def __init__(self, n_features, n_hidden, fine_tune_resnet=False):
        super(VisualRefListenerOracle, self).__init__()
        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.fc1 = nn.Linear(resnet.fc.in_features, n_features)
        self.fc2 = nn.Linear(n_features, n_hidden)

    def forward(self, rnn_out, receiver_input):
        batch_size = receiver_input[0].shape[0]

        images_1, images_2 = receiver_input

        emb_1 = self.resnet(images_1)
        emb_1 = emb_1.view(emb_1.size(0), -1)
        emb_1 = self.fc1(emb_1)
        # TODO add nonlinearity?
        emb_1 = self.fc2(emb_1)

        emb_2 = self.resnet(images_2)
        emb_2 = emb_2.view(emb_2.size(0), -1)
        emb_2 = self.fc1(emb_2)
        emb_2 = self.fc2(emb_2)

        stacked = torch.stack([emb_1, emb_2], dim=1)

        # TODO correct like this?
        dots = torch.matmul(stacked, torch.unsqueeze(rnn_out, dim=-1))
        return dots.view(batch_size, -1)
