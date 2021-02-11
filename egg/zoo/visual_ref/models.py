import os
import pickle

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torchtext.vocab import Vocab
from torchvision.models import resnet50
from torch.autograd import Variable

import numpy as np

from egg.zoo.visual_ref.preprocess import TOKEN_START, TOKEN_END
from egg.zoo.visual_ref.utils import SPECIAL_CHARACTERS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images_embedded, captions_embedded):
        # compute image-caption score matrix
        scores = cosine_sim(images_embedded, captions_embedded)
        diagonal = scores.diag().view(images_embedded.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # Sum up caption retrieval and image retrieval loss
        sum_of_losses = cost_s.sum() + cost_im.sum()

        # Normalize loss by batch size
        normalized_loss = sum_of_losses / images_embedded.size(0)

        return normalized_loss


def cosine_sim(images_embedded, captions_embedded):
    """Cosine similarity between all the image and sentence pairs."""
    return images_embedded.mm(captions_embedded.t())


class ImageSentenceRanker(nn.Module):
    def __init__(self, word_embedding_size, joint_embeddings_size, lstm_hidden_size, vocab_size, fine_tune_resnet=True):
        super(ImageSentenceRanker, self).__init__()
        self.image_embedding = ImageEmbedding(
            joint_embeddings_size, fine_tune_resnet
        )
        self.caption_embedding = nn.Linear(
            lstm_hidden_size,
            joint_embeddings_size,
        )

        #TODO no word embeddings used in paper?
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_size)

        self.language_encoding_lstm = LanguageEncodingLSTM(
            word_embedding_size,
            lstm_hidden_size,
        )

        self.lstm_hidden_size = lstm_hidden_size

        self.loss = ContrastiveLoss()

    def accuracy_discrimination(self, images_embedded, captions_embedded):
        """Calculate accuracy of model when discriminating 2 images/captions"""
        accuracies = []
        for i, (image_1_embedded, caption_1_embedded) in enumerate(zip(images_embedded, captions_embedded)):
            for j, (image_2_embedded, caption_2_embedded) in enumerate(zip(images_embedded, captions_embedded)):
                # disregard cases where images are the same
                if i != j:
                    similarities = cosine_sim(torch.stack((image_1_embedded, image_2_embedded)), torch.stack((caption_1_embedded, caption_2_embedded)))
                    # caption_0 is more similar to image_0 than to image_1
                    if similarities[0, 0] > similarities[0, 1]:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)

                    # caption_1 is more similar to image_1 than to image_0
                    if similarities[1, 1] > similarities[1, 0]:
                        accuracies.append(1)
                    else:
                        accuracies.append(0)

        return np.mean(accuracies)

    def embed_captions(self, captions, decode_lengths):
        # Initialize LSTM state
        batch_size = captions.size(0)
        h_lan_enc, c_lan_enc = self.language_encoding_lstm.init_state(batch_size)

        # TODO use packed sequences

        # Tensor to store hidden activations
        lang_enc_hidden_activations = torch.zeros(
            (batch_size, self.lstm_hidden_size), device=device
        )

        for t in range(max(decode_lengths)):
            prev_words_embedded = self.word_embedding(captions[:, t])

            h_lan_enc, c_lan_enc = self.language_encoding_lstm(
                h_lan_enc, c_lan_enc, prev_words_embedded
            )

            lang_enc_hidden_activations[decode_lengths == t + 1] = h_lan_enc[
                decode_lengths == t + 1
            ]

        captions_embedded = self.caption_embedding(lang_enc_hidden_activations)
        captions_embedded = l2_norm(captions_embedded)
        return captions_embedded

    def forward(self, encoder_output, captions, caption_lengths):
        """
        Forward propagation for the ranking task.
        """
        images_embedded = self.image_embedding(encoder_output)
        captions_embedded = self.embed_captions(captions, caption_lengths)

        return images_embedded, captions_embedded


class LanguageEncodingLSTM(nn.Module):
    def __init__(self, word_embeddings_size, hidden_size):
        super(LanguageEncodingLSTM, self).__init__()
        self.lstm_cell = nn.LSTMCell(word_embeddings_size, hidden_size)

    def forward(self, h, c, prev_words_embedded):
        h_out, c_out = self.lstm_cell(prev_words_embedded, (h, c))
        return h_out, c_out

    def init_state(self, batch_size):
        h = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        c = torch.zeros((batch_size, self.lstm_cell.hidden_size), device=device)
        return [h, c]


def l2_norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class ImageEmbedding(nn.Module):
    def __init__(self, joint_embeddings_size, fine_tune_resnet):
        super(ImageEmbedding, self).__init__()

        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, joint_embeddings_size)

    def forward(self, images):
        images_embedded = self.resnet(images)
        images_embedded = self.embed(images_embedded.squeeze())

        return images_embedded


class ImageCaptioner(nn.Module):
    def __init__(self, word_embedding_size, visual_embedding_size, lstm_hidden_size, vocab, max_caption_length,
                 fine_tune_resnet=True):
        super(ImageCaptioner, self).__init__()

        resnet = resnet50(pretrained=True)

        if not fine_tune_resnet:
            for param in resnet.parameters():
                param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, visual_embedding_size)

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
        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.embed(image_features)

        batch_size = images.shape[0]

        hidden = self.init_hidden(batch_size)

        # cut off <eos> token as it's not needed to predict with this as input
        captions = captions[:, :-1]
        caption_lengths = [l-1 for l in caption_lengths]

        embedded_captions = self.word_embedding(captions)

        image_features = image_features.unsqueeze(dim=1).repeat(1, embedded_captions.shape[1], 1)

        inputs = torch.cat((image_features, embedded_captions), dim=2)
        packed_inputs = pack_padded_sequence(inputs, caption_lengths, enforce_sorted=False, batch_first=True)

        lstm_out, hidden = self.lstm(packed_inputs, hidden)
        output, _ = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.fc(output)
        output = output.transpose(1, 2)
        return output


    def forward_decode(self, images, decode_type="greedy"):
        """ Forward propagation at test time (no teacher forcing)."""

        if decode_type not in ["greedy", "sample"]:
            raise NotImplementedError("Unknown decode type: ", decode_type)

        image_features = self.resnet(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.embed(image_features)

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
            if decode_type == "greedy":
                # greedy decode
                prev_words = torch.argmax(scores_for_timestep, dim=1)
            else:
                # sample from the distribution
                prev_words = torch.multinomial(torch.softmax(scores_for_timestep, -1), 1).squeeze()

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


class VisualRefSpeakerDiscriminativeOracle(nn.Module):
    def __init__(self, data_folder, captions_filename, max_sequence_length, vocab):
        super(VisualRefSpeakerDiscriminativeOracle, self).__init__()

        # Load captions
        with open(os.path.join(data_folder, captions_filename["train"]), "rb") as file:
            self.captions_train = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["val"]), "rb") as file:
            self.captions_val = pickle.load(file)

        with open(os.path.join(data_folder, captions_filename["test"]), "rb") as file:
            self.captions_test = pickle.load(file)

        self.max_sequence_length = max_sequence_length

        self.vocab = vocab
        self.ignore_tokens = [vocab.stoi[s] for s in [Vocab.UNK] + SPECIAL_CHARACTERS]

    def pad_messages(self, messages, padding_value=0.0):
        """Trim and pad all messages to max sequence length."""
        trailing_dims = messages[0].size()[1:]
        max_len = self.max_sequence_length
        out_dims = (len(messages), max_len) + trailing_dims

        out_tensor = messages[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(messages):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            out_tensor[i, :length, ...] = tensor[:self.max_sequence_length]

        return out_tensor

    def forward(self, input):
        # input: target_image, distractor_image
        images, target_label, target_image_ids, distractor_image_ids = input

        # pick the target's caption that has the least word overlap with any of the distractor's captions
        # (the score is normalized by the captions’ length excluding stop-words)
        def get_relevant_tokens(caption):
            return set([t for t in caption if t not in self.ignore_tokens])

        if self.training:
            captions = self.captions_train
        else:
            # TODO test mode
            captions = self.captions_val

        output_captions = []
        for target_image_id, distractor_image_id in zip(target_image_ids, distractor_image_ids):
            overlap_scores = []
            for target_caption in captions[int(target_image_id)]:
                overlap_score = 0
                target_caption_tokens = get_relevant_tokens(target_caption)

                distractor_captions = [t for caption in captions[int(distractor_image_id)] for t in caption]
                distractor_captions_tokens = get_relevant_tokens(distractor_captions)

                overlap = len(target_caption_tokens & distractor_captions_tokens)
                overlap_scores.append(overlap / len(target_caption_tokens))

            best_caption_idx = np.argmin(overlap_scores)
            best_caption = captions[int(target_image_id)][best_caption_idx]
            output_captions.append(best_caption)

        # append end of message token
        for caption in output_captions:
            caption.append(0)

        # Transform lists to tensors
        output_captions = [torch.tensor(caption, device=device) for caption in output_captions]

        # Pad all captions in batch to equal length
        output_captions = self.pad_messages(output_captions)

        # out: sender RNN init hidden state
        # out: sequence, logits, entropy
        return output_captions, None, None


class VisualRefListenerOracle(nn.Module):
    def __init__(self, ranking_model):
        super(VisualRefListenerOracle, self).__init__()
        self.ranking_model = ranking_model

    def forward(self, messages, receiver_input, message_lengths):
        batch_size = receiver_input[0].shape[0]

        images_1, images_2 = receiver_input

        images_1_embedded, messages_embedded = self.ranking_model(images_1, messages, message_lengths)
        images_2_embedded, messages_embedded = self.ranking_model(images_2, messages, message_lengths)

        stacked_images = torch.stack([images_1_embedded, images_2_embedded], dim=1)

        similarities = torch.matmul(stacked_images, torch.unsqueeze(messages_embedded, dim=-1))

        # out: scores, logits, entropy
        return similarities.view(batch_size, -1),  None, None
