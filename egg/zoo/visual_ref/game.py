from collections import defaultdict
from typing import Callable

import torch
from torch import nn

from egg.core import LoggingStrategy, find_lengths
from egg.core.baselines import Baseline, MeanBaseline
from egg.core.reinforce_wrappers import _verify_batch_sizes


class OracleSenderReceiverRnnReinforce(nn.Module):
    """
    Implements Sender/Receiver game with training done via Reinforce. Both agents are supposed to
    return 3-tuples of (output, log-prob of the output, entropy).
    The game implementation is responsible for handling the end-of-sequence term, so that the optimized loss
    corresponds either to the position of the eos term (assumed to be 0) or the end of sequence.

    Sender and Receiver can be obtained by applying the corresponding wrappers.
    `SenderReceiverRnnReinforce` also applies the mean baseline to the loss function to reduce
    the variance of the gradient estimate.

    >>> sender = nn.Linear(3, 10)
    >>> sender = RnnSenderReinforce(sender, vocab_size=15, embed_dim=5, hidden_size=10, max_len=10, cell='lstm')

    >>> class Receiver(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc = nn.Linear(5, 3)
    ...     def forward(self, rnn_output, _input = None):
    ...         return self.fc(rnn_output)
    >>> receiver = RnnReceiverDeterministic(Receiver(), vocab_size=15, embed_dim=10, hidden_size=5)
    >>> def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
    ...     loss = F.mse_loss(sender_input, receiver_output, reduction='none').mean(dim=1)
    ...     aux = {'aux': torch.ones(sender_input.size(0))}
    ...     return loss, aux
    >>> game = SenderReceiverRnnReinforce(sender, receiver, loss, sender_entropy_coeff=0.0, receiver_entropy_coeff=0.0,
    ...                                   length_cost=1e-2)
    >>> input = torch.zeros((5, 3)).normal_()
    >>> optimized_loss, interaction = game(input, labels=None)
    >>> sorted(list(interaction.aux.keys()))  # returns some debug info, such as entropies of the agents, msg length etc
    ['aux', 'length', 'receiver_entropy', 'sender_entropy']
    >>> interaction.aux['aux'], interaction.aux['aux'].sum()
    (tensor([1., 1., 1., 1., 1.]), tensor(5.))
    """

    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        sender_entropy_coeff: float = 0.0,
        receiver_entropy_coeff: float = 0.0,
        length_cost: float = 0.0,
        baseline_type: Baseline = MeanBaseline,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        """
        :param sender: sender agent
        :param receiver: receiver agent
        :param loss:  the optimized loss that accepts
            sender_input: input of Sender
            message: the is sent by Sender
            receiver_input: input of Receiver from the dataset
            receiver_output: output of Receiver
            labels: labels assigned to Sender's input data
          and outputs a tuple of (1) a loss tensor of shape (batch size, 1) (2) the dict with auxiliary information
          of the same shape. The loss will be minimized during training, and the auxiliary information aggregated over
          all batches in the dataset.

        :param sender_entropy_coeff: entropy regularization coeff for sender
        :param receiver_entropy_coeff: entropy regularization coeff for receiver
        :param length_cost: the penalty applied to Sender for each symbol produced
        :param baseline_type: Callable, returns a baseline instance (eg a class specializing core.baselines.Baseline)
        :param train_logging_strategy, test_logging_strategy: specify what parts of interactions to persist for
            later analysis in callbacks

        """
        super(OracleSenderReceiverRnnReinforce, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.sender_entropy_coeff = sender_entropy_coeff
        self.receiver_entropy_coeff = receiver_entropy_coeff
        self.loss = loss
        self.length_cost = length_cost

        self.baselines = defaultdict(baseline_type)
        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None):
        message, log_prob_s, entropy_s = self.sender(sender_input)
        message_length = find_lengths(message)
        receiver_output, log_prob_r, entropy_r = self.receiver(
            message, receiver_input, message_length
        )

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels
        )

        # TODO fix
        # if self.training:
        #     _verify_batch_sizes(loss, log_prob_s, log_prob_r)

        # the entropy of the outputs of S before and including the eos symbol - as we don't care about what's after
        effective_entropy_s = torch.zeros_like(entropy_r)

        # the log prob of the choices made by S before and including the eos symbol - again, we don't
        # care about the rest
        effective_log_prob_s = torch.zeros_like(log_prob_r)

        # for i in range(message.size(1)):
        #     not_eosed = (i < message_length).float()
        #     effective_entropy_s += entropy_s[:, i] * not_eosed
        #     effective_log_prob_s += log_prob_s[:, i] * not_eosed
        # effective_entropy_s = effective_entropy_s / message_length.float()

        # weighted_entropy = (
        #     effective_entropy_s.mean() * self.sender_entropy_coeff
        #     + entropy_r.mean() * self.receiver_entropy_coeff
        # )
        weighted_entropy = (
                entropy_r.mean() * self.receiver_entropy_coeff
        )

        # log_prob = effective_log_prob_s + log_prob_r

        # length_loss = message_length.float() * self.length_cost
        #
        # policy_length_loss = (
        #     (length_loss - self.baselines["length"].predict(length_loss))
        #     * effective_log_prob_s
        # ).mean()  # noqa: E502
        # policy_loss = (
        #     (loss.detach() - self.baselines["loss"].predict(loss.detach())) * log_prob
        # ).mean()
        #
        # optimized_loss = policy_length_loss + policy_loss - weighted_entropy
        # if the receiver is deterministic/differentiable, we apply the actual loss
        # optimized_loss += loss.mean()
        # TODO: add entropy regularization term
        optimized_loss = loss.mean()

        if self.training:
            self.baselines["loss"].update(loss)
            # self.baselines["length"].update(length_loss)

        # aux_info["sender_entropy"] = entropy_s.detach()
        aux_info["receiver_entropy"] = entropy_r.detach()
        aux_info["length"] = message_length.float()  # will be averaged

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            message=message.detach(),
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )

        return optimized_loss, interaction