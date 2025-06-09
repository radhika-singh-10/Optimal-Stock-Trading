# dqn_pytorch.py
import time
import math
import numpy as np
import pandas as pd
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple(
    "Transition",
    ["state0", "state1", "reward"]
)


class PrioritisedReplay:
    """
    Rough PyTorch equivalent of your `SequentialMemory`.
    Priority is proportional to |TD-error|; sampling is roulette-wheel.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.eps = 1e-6       # small constant to avoid zero prob.

    def __len__(self):
        return len(self.memory)

    def append(self, state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)

        max_p = max(self.priorities) if self.priorities else 1.0
        self.memory[self.pos] = state
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    # --------------------------------------------------------------------- #
    def sample(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples")

        # probability ∝ p_i ** α
        scaled_p = np.asarray(self.priorities, dtype=np.float32) ** alpha
        probs = scaled_p / scaled_p.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # importance-sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalise to 1

        # Build transitions
        state0 = torch.from_numpy(np.stack(samples[:-1])).float()
        state1 = torch.from_numpy(np.stack(samples[1:])).float()
        reward = torch.from_numpy(np.stack([s[:, :, -1] for s in samples])).float()

        return Transition(state0, state1, reward), torch.from_numpy(weights).float(), indices

    def update_priority(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = float(abs(err) + self.eps)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  Multi-Scale CNN block (critic network)                                     #
# --------------------------------------------------------------------------- #
class MultiScaleCritic(nn.Module):
    """
    Direct PyTorch port of build_critic():  raw -> conv,   smoothed -> conv,
    down-sampled -> conv,   then concatenate (channel dim) and FC head.
    """

    def __init__(
        self,
        n_stock: int,
        history_length: int,
        n_smooth: int,
        n_down: int,
        k_w: int,
        n_feature: int,
        n_hidden: int,
    ):
        super().__init__()

        self.history_length = history_length
        self.n_smooth = n_smooth
        self.n_down = n_down

        # One conv block we can share (identical hyper-params)
        def _make_conv_block(in_channels=1, out_channels=n_feature):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(k_w, 1),
                    padding=(k_w // 2, 0),
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )

        # raw stream
        self.raw_net = _make_conv_block()

        # smoothed + downsampled streams (lists of blocks)
        self.sm_nets = nn.ModuleList([_make_conv_block() for _ in range(n_smooth)])
        self.dw_nets = nn.ModuleList([_make_conv_block() for _ in range(n_down)])

        # second shared conv after concat
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                n_feature * (1 + n_smooth + n_down),
                n_feature * 2,
                kernel_size=(k_w, 1),
                padding=(k_w // 2, 0),
            ),
            nn.BatchNorm2d(n_feature * 2),
            nn.PReLU(),
        )

        # FC head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((n_feature * 2) * history_length * n_stock, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.PReLU(),
            nn.Linear(n_hidden, int(math.sqrt(n_hidden))),
            nn.PReLU(),
            nn.Linear(int(math.sqrt(n_hidden)), 2 * n_stock),
        )

        self.n_stock = n_stock

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_stock)
        Returns: (B, n_stock, 2)  (Q-values for exit / stay)
        """
        B, T, n_stock = x.shape
        assert n_stock == self.n_stock, "n_stock mismatch"

        # switch to (B, 1, T, n_stock)  (CHW for Conv2d)
        x = x.unsqueeze(1)

        # ----------------------------- raw ------------------------------- #
        raw = x[:, :, -self.history_length :, :]  # latest history

        # -------------------------- smoothed ---------------------------- #
        smoothed_out = []
        for n_sm, net in enumerate(self.sm_nets, start=2):
            # average across n_sm rolling windows
            segments = []
            for st in range(n_sm):
                start = -(self.history_length + st)
                end = -st if st != 0 else None
                segments.append(x[:, :, start:end, :])
            stacked = torch.stack(segments, dim=0).mean(0)
            smoothed_out.append(net(stacked))

        # ------------------------ down-sampled -------------------------- #
        down_out = []
        for n_dw, net in enumerate(self.dw_nets, start=2):
            idx = torch.arange(
                T - n_dw * self.history_length, T, n_dw, device=x.device
            )
            downsampled = x.index_select(2, idx)
            down_out.append(net(downsampled))

        # run raw conv
        raw_out = self.raw_net(raw)

        # concat along channel (dim=1)
        merged = torch.cat([raw_out] + smoothed_out + down_out, dim=1)

        y = self.conv2(merged)
        y = self.fc(y)
        return y.view(B, self.n_stock, 2)


# --------------------------------------------------------------------------- #
#                               DQN wrapper                                   #
# --------------------------------------------------------------------------- #
class DQNPytorchWrapper:
    """
    High-level wrapper for DQN.  
    (build_model(), train(), predict_action() …) but drops TF.
    """

    def __init__(self, config):

        # copy simple scalars
        for k, v in config.__dict__.items():
            setattr(self, k, v)

        # derived
        self.n_history = max(
            self.n_smooth + self.history_length,
            (self.n_down + 1) * self.history_length,
        )

        # networks
        self.device = torch.device(self.device)
        self.online = MultiScaleCritic(
            n_stock=self.n_stock,
            history_length=self.history_length,
            n_smooth=self.n_smooth,
            n_down=self.n_down,
            k_w=self.k_w,
            n_feature=self.n_feature,
            n_hidden=self.n_hidden,
        ).to(self.device)

        self.target_net = MultiScaleCritic(
            n_stock=self.n_stock,
            history_length=self.history_length,
            n_smooth=self.n_smooth,
            n_down=self.n_down,
            k_w=self.k_w,
            n_feature=self.n_feature,
            n_hidden=self.n_hidden,
        ).to(self.device)

        self.target_net.load_state_dict(self.online.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.learning_rate)
        self.memory = [PrioritisedReplay(self.memory_length) for _ in range(self.n_memory)]
        self.mse = nn.MSELoss(reduction="none")  # we need TD-errors per sample

    # ----------------------------------------------------------------------- #
    #                              Public API                                 #
    # ----------------------------------------------------------------------- #
    def predict_action(self, state: np.ndarray):
        """
        state: shape (n_stock,)
        Returns numpy int array (n_stock,) with 0-exit / 1-stay.
        """
        self.online.eval()
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0)  # (1, n_stock)
            # need a synthetic batch of length self.n_history to feed the net
            # here we just replicate last value
            hist = st.repeat(self.n_history, 1).unsqueeze(0)  # (1, T, n_stock)
            q = self.online(hist.to(self.device))
            act = q.argmax(-1).cpu().numpy()[0]
        return act
    

    # ----------------------------------------------------------------------- #
    def train(self, input_data: pd.DataFrame, noise_scale: float = 0.1):
        """
        Port of your long train() method – condensed and PyTorch-ified.
        You may want to adapt logging / checkpoints.
        """
        stock_data = input_data.values.astype(np.float32)
        dates = input_data.index
        T = len(stock_data)

        # warm-up memory
        print("--> Initialising replay buffers")
        for t in range(self.n_history):
            for m in self.memory:
                m.append(stock_data[t], reward=np.zeros((self.n_stock, 2)))

        print("--> Start training")
        for t in range(self.n_history, T):
            state_t = stock_data[t]
            # ε-greedy exploration
            epsilon = max(0.05, 1.0 - t / (0.5 * T))
            if np.random.rand() < epsilon:
                action = np.random.randint(0, 2, size=self.n_stock)
            else:
                action = self.predict_action(state_t)

            # reward structure identical to TF version
            reward = np.concatenate(
                [state_t.reshape(self.n_stock, 1), np.zeros((self.n_stock, 1))], axis=-1
            )

            # save transition
            for m in self.memory:
                m.append(state_t, reward)

            # ---------------------------------------------------------------- #
            #  optimisation step                                              #
            # ---------------------------------------------------------------- #
            for _ in range(self.n_epochs):
                idx_bank = np.random.randint(0, self.n_memory)
                batch, weights, indices = self.memory[idx_bank].sample(
                    self.n_batch, alpha=self.alpha, beta=self.beta
                )

                # move to device
                s0 = batch.state0.to(self.device)
                s1 = batch.state1.to(self.device)
                r = batch.reward.to(self.device)
                w = weights.to(self.device)

                # Q(s1, ·) from target net
                with torch.no_grad():
                    q_next = self.target_net(s1).max(-1, keepdim=True)[0]
                    target = r + self.gamma * torch.cat(
                        [torch.zeros_like(q_next), q_next], dim=-1
                    )

                # current Q
                self.online.train()
                q_now = self.online(s0)

                # loss per sample
                td_error = (q_now - target).abs().detach().mean((1, 2)).cpu().numpy()
                self.memory[idx_bank].update_priority(indices, td_error)

                loss = (w * self.mse(q_now, target).mean((1, 2))).mean()

                # gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
                self.optimizer.step()

            # ---------------------- soft update target ----------------------- #
            tau = self.update_rate
            for online_p, target_p in zip(
                self.online.parameters(), self.target_net.parameters()
            ):
                target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)

            # ---------------------- logging each ~1 % ------------------------ #
            if t % max(1, T // 100) == 0:
                print(
                    f"{dates[t].date()}  step {t:>6}/{T}  "
                    f"ε={epsilon:.3f}  loss={loss.item():.4f}"
                )

        print("Training finished ✔")
        # could return final exit dates the same way as in original code
# --------------------------------------------------------------------------- #