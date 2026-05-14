from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import typing

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi

    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )

    return embedding


# =========================
# Boundary-Constrained Flow Model
# =========================
class FlowModel(nn.Module):
    def __init__(
        self,
        dims: typing.List,
        time_emb_size: int,
        time_type="cat",
        act_func="tanh",
        norm=False,
        init_dropout=0.0,
        dropout=0.1,
        sigma_data=0.5,
    ):
        super(FlowModel, self).__init__()

        # Avoid modifying the original dims list by reference.
        self.dims = dims.copy()

        self.time_type = time_type
        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.sigma_data = sigma_data

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError("Unimplemented timestep embedding type")

        self.mlp_layers = MLPLayers(
            layers=self.dims,
            dropout=dropout,
            activation=act_func,
            last_activation=False,
        )

        self.init_dropout = nn.Dropout(init_dropout)
        self.apply(xavier_normal_initialization)

    def c_skip(self, t):
        tau = 1.0 - t
        return self.sigma_data ** 2 / (tau ** 2 + self.sigma_data ** 2)

    def c_out(self, t):
        tau = 1.0 - t
        return tau * self.sigma_data / torch.sqrt(tau ** 2 + self.sigma_data ** 2)

    def backbone(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x, dim=-1)

        x_drop = self.init_dropout(x)
        h = torch.cat([x_drop, emb], dim=-1)

        return self.mlp_layers(h)

    def forward(self, x, t):
        net_out = self.backbone(x, t)

        c_skip = self.c_skip(t).unsqueeze(-1)
        c_out = self.c_out(t).unsqueeze(-1)

        return c_skip * x + c_out * net_out


# =========================
# ConsCF with EMA + Consistency + Masked History Prior
# =========================
class ConsCF(GeneralRecommender, AutoEncoderMixin):
    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(ConsCF, self).__init__(config, dataset)
        super().build_histroy_items(dataset)

        # =========================
        # Basic config
        # =========================
        self.n_steps = config["n_steps"]
        self.s_steps = config["s_steps"] if "s_steps" in config else 1

        # Register as buffer so it is moved together with the model.
        self.register_buffer("time_steps", torch.linspace(0, 1, self.n_steps + 1))

        self.time_emb_size = config["time_embedding_size"]
        dims = [self.n_items] + config["dims_mlp"] + [self.n_items]

        # =========================
        # EMA, consistency, and boundary config
        # =========================
        self.ema_decay = config["ema_decay"] if "ema_decay" in config else 0.999

        # Fixed consistency weight.
        # It is no longer read from config and should not be swept
        # for sensitivity analysis.
        self.lambda_cons = 1.0

        self.sigma_data = config["sigma_data"] if "sigma_data" in config else 0.5

        # =========================
        # Masked History Prior config
        # x0 = m * x1, m_i ~ Bernoulli(prior_keep_prob)
        # =========================
        self.prior_keep_prob = (
            config["prior_keep_prob"] if "prior_keep_prob" in config else 0.2
        )

        # If true, each user keeps at least one interacted item when possible.
        # This avoids an all-zero x0 for users with very short histories.
        self.prior_ensure_nonzero = (
            config["prior_ensure_nonzero"]
            if "prior_ensure_nonzero" in config
            else True
        )

        # Whether to focus training timesteps on the same interval used in inference.
        # Keeping this false is the more conservative default.
        self.align_train_infer_steps = (
            config["align_train_infer_steps"]
            if "align_train_infer_steps" in config
            else False
        )

        # Whether to use EMA model for full-sort prediction.
        self.use_ema_predict = (
            config["use_ema_predict"] if "use_ema_predict" in config else False
        )

        # Whether to mask observed items inside full_sort_predict.
        # RecBole evaluation usually handles this, so the default is False.
        self.mask_seen_in_predict = (
            config["mask_seen_in_predict"]
            if "mask_seen_in_predict" in config
            else False
        )
        self.flow_model = FlowModel(
            dims=dims,
            time_emb_size=self.time_emb_size,
            dropout=config["dropout"] if "dropout" in config else 0.1,
            init_dropout=config["init_dropout"] if "init_dropout" in config else 0.0,
            sigma_data=self.sigma_data,
        )
        self.flow_model_ema = FlowModel(
            dims=dims,
            time_emb_size=self.time_emb_size,
            dropout=config["dropout"] if "dropout" in config else 0.1,
            init_dropout=config["init_dropout"] if "init_dropout" in config else 0.0,
            sigma_data=self.sigma_data,
        )

        # Initialize teacher = student.
        self.flow_model_ema.load_state_dict(self.flow_model.state_dict())

        for p in self.flow_model_ema.parameters():
            p.requires_grad = False
    @torch.no_grad()
    def update_ema(self):
        for p_ema, p in zip(
            self.flow_model_ema.parameters(), self.flow_model.parameters()
        ):
            p_ema.data.mul_(self.ema_decay).add_(
                p.data, alpha=1.0 - self.ema_decay
            )
    @torch.no_grad()
    def sample_behavior_prior(self, x1):
        keep_prob = float(self.prior_keep_prob)
        keep_prob = max(0.0, min(1.0, keep_prob))

        keep_mask = torch.rand_like(x1, dtype=torch.float32) < keep_prob
        x0 = x1 * keep_mask.float()

        # Item 0 is usually the padding item in RecBole.
        x0[:, 0] = 0.0

        if self.prior_ensure_nonzero:
            x0 = self._ensure_nonzero_prior(x0, x1)

        return x0

    @torch.no_grad()
    def _ensure_nonzero_prior(self, x0, x1):
        x1_degree = x1.sum(dim=1)
        x0_degree = x0.sum(dim=1)

        need_fix = (x1_degree > 0) & (x0_degree <= 0)
        if not torch.any(need_fix):
            return x0

        rows = torch.where(need_fix)[0]

        for row in rows:
            pos_items = torch.where(x1[row] > 0)[0]
            pos_items = pos_items[pos_items > 0]

            if pos_items.numel() == 0:
                continue

            sampled_idx = torch.randint(
                low=0,
                high=pos_items.numel(),
                size=(1,),
                device=x1.device,
            )
            item = pos_items[sampled_idx]
            x0[row, item] = 1.0

        return x0

    def forward(self, x, t):
        return self.flow_model(x, t)
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user).float()

        batch_size = x1.size(0)
        if self.align_train_infer_steps:
            start_step = max(0, self.n_steps - self.s_steps)
        else:
            start_step = 0
        steps = torch.randint(
            start_step,
            self.n_steps,
            (batch_size,),
            device=x1.device,
        )
        t = self.time_steps[steps].to(x1.device).unsqueeze(1)
        t_next = self.time_steps[steps + 1].to(x1.device).unsqueeze(1)
        x0 = self.sample_behavior_prior(x1)
        u = torch.rand_like(x1, dtype=torch.float32)
        mask_t = u <= t
        mask_next = u <= t_next
        xt = torch.where(mask_t, x1, x0)
        xt_next = torch.where(mask_next, x1, x0)
        xt[:, 0] = 0.0
        xt_next[:, 0] = 0.0
        pred = self.flow_model(xt, t.squeeze(-1))
        with torch.no_grad():
            teacher_pred = self.flow_model_ema(
                xt_next,
                t_next.squeeze(-1),
            )
        loss_rec = mean_flat((x1 - pred) ** 2)
        loss_cons = mean_flat((pred - teacher_pred) ** 2)

        loss = loss_rec.mean() + self.lambda_cons * loss_cons.mean()

        self.update_ema()

        return loss
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        X_bar = self.get_rating_matrix(user).float()

        model = self.flow_model_ema if self.use_ema_predict else self.flow_model
        i_t = self.n_steps - 1
        t = self.time_steps[i_t].repeat(X_bar.shape[0], 1).to(X_bar.device)

        scores = model(X_bar, t.squeeze(-1))

        if self.mask_seen_in_predict:
            scores = scores.masked_fill(X_bar > 0, -1e9)
            scores[:, 0] = -1e9

        return scores

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        return x_t[:, item]
