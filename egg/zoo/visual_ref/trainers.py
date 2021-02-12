import os
import pathlib
from typing import List, Optional

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from egg.core import Trainer, Callback, Interaction, move_to, get_opts


class VisualRefTrainer(Trainer):
    """
    Implements the training logic. Some common configuration (checkpointing frequency, path, validation frequency)
    is done by checking util.common_opts that is set via the CL.
    """

    def __init__(
        self,
        game: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: DataLoader,
        validation_data: Optional[DataLoader] = None,
        device: torch.device = None,
        callbacks: Optional[List[Callback]] = None,
        grad_norm: float = None,
        aggregate_interaction_logs: bool = True,
    ):
        super(VisualRefTrainer, self).__init__(game, optimizer, train_data, validation_data, device, callbacks, grad_norm, aggregate_interaction_logs)
        common_opts = get_opts()
        self.eval_frequency = common_opts.eval_frequency

    def train_epoch(self):
        mean_loss = 0
        n_batches = 0
        interactions = []

        self.game.train()

        self.optimizer.zero_grad()

        for batch_id, batch in enumerate(self.train_data):
            batch = move_to(batch, self.device)

            context = autocast() if self.scaler else nullcontext()
            with context:
                optimized_loss, interaction = self.game(*batch)

                if self.update_freq > 1:
                    # throughout EGG, we minimize _mean_ loss, not sum
                    # hence, we need to account for that when aggregating grads
                    optimized_loss = optimized_loss / self.update_freq

            if self.scaler:
                self.scaler.scale(optimized_loss).backward()
            else:
                optimized_loss.backward()

            if batch_id % self.update_freq == self.update_freq - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                if self.grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.game.parameters(), self.grad_norm
                    )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            n_batches += 1
            mean_loss += optimized_loss.detach()
            if (
                self.distributed_context.is_distributed
                and self.aggregate_interaction_logs
            ):
                interaction = Interaction.gather_distributed_interactions(interaction)
            interaction = interaction.to("cpu")

            for callback in self.callbacks:
                callback.on_batch_end(interaction, optimized_loss, batch_id)

            interactions.append(interaction)

            if batch_id % self.eval_frequency == self.eval_frequency - 1:
                val_loss, interactions = self.eval()
                print(f"Batch {batch_id} | Val loss: {val_loss:.3f} | Val acc: {interactions.aux['acc'].mean():.3f}\n")

        mean_loss /= n_batches
        full_interaction = Interaction.from_iterable(interactions)
        return mean_loss.item(), full_interaction

