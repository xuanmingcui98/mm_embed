from torch import Tensor
import torch.distributed as dist
import torch
import torch.nn.functional as F


class SimpleContrastiveLoss:
    def __init__(self, use_symmetric_loss: bool = False):
        self.use_symmetric_loss = use_symmetric_loss

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        hard_neg: Tensor | None = None,   # shape [N, K, D] (2-D mode) or [B, K, D] (late interaction)
        x_task_ids: Tensor = None,
        y_task_ids: Tensor = None,
        target: Tensor = None,
        temperature: float = 0.02,
        inter_task_temperature: float | None = None,
        reduction: str = 'mean',
        # ---- hard negatives (optional) ----
        hard_neg_weight: float = 1.0,     # λ in the denominator
    ) -> Tensor:
        """
        If `hard_neg` is provided, we append λ * exp(sim(q_i, hard_k)/τ) to each query's denominator.
        Notes:
          - Inter-task temperature adjustments currently *do not* apply to hard negatives.
          - When `use_symmetric_loss=True`, hard negatives are only applied on x->y direction.
        """
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long
            )

        # ----- compute logits(q, y) -----
        if x.dim() == 3 and y.dim() == 3:
            # late interaction: x[b, n_q, d], y[c, s, d] -> logits[b, c, n_q, s]
            logits = torch.einsum("bnd,csd->bcns", x, y)
            # aggregate to [b, c] like your original rule (amax over s, avg over n_q)
            logits = logits.amax(dim=3).sum(dim=2) / x.shape[1]
        else:
            # standard 2-D case: [N, D] x [M, D]^T -> [N, M]
            logits = torch.matmul(x, y.transpose(0, 1))

        # ----- temperature (may be scalar or pairwise if inter-task provided) -----
        T = temperature
        if inter_task_temperature is not None and x_task_ids is not None and y_task_ids is not None:
            same_task = x_task_ids.unsqueeze(1) == y_task_ids.unsqueeze(0)  # [N, M]
            # if late interaction, shapes are [B, C] which still works
            T = torch.where(same_task, torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype),
                            torch.as_tensor(inter_task_temperature, device=logits.device, dtype=logits.dtype))

        scaled_logits = logits / T

        # ----- hard negatives: compute logits(q, hard_neg) and append to denominator -----
        # hard_neg is per-query: shape [N, K, D] (or [B, K, D] in late interaction)
        # Produce hard_logits of shape [N, K] (or [B, K]) and then scale by τ.
        hard_logits = None
        if hard_neg is not None:
            if x.dim() == 3:
                # late interaction: x[b, n_q, d], hard_neg[b, K, d] -> [b, n_q, K]
                hn_pair = torch.einsum("bnd,bkd->bnk", x, hard_neg)  # dot per (n_q, K)
                # aggregate across n_q like above (avg over queries)
                hard_logits = hn_pair.sum(dim=1) / x.shape[1]        # [b, K]
            else:
                # 2-D case: x[N, D], hard_neg[N, K, D]
                # einsum to [N, K]
                hard_logits = torch.einsum("nd,nkd->nk", x, hard_neg)

            # scale by the (scalar) temperature; inter-task temperature not applied to hard negs
            hard_logits = hard_logits / temperature  # [*, K]

        # ----- loss -----
        if self.use_symmetric_loss:
            # Symmetric loss: avg of CE(x->y) and CE(y->x).
            # We only include hard negatives on x->y side to keep it simple (as requested).
            if hard_logits is not None:
                # Concatenate hard negatives to the denominator for x->y
                denom_logits_xy = torch.cat([scaled_logits, hard_neg_weight * hard_logits], dim=1)  # [N, M+K]
                loss_x = F.cross_entropy(denom_logits_xy, target, reduction=reduction)
            else:
                loss_x = F.cross_entropy(scaled_logits, target, reduction=reduction)

            loss_y = F.cross_entropy(scaled_logits.t(), target, reduction=reduction)
            loss = (loss_x + loss_y) / 2
        else:
            # Single-direction (x->y) with optional hard negatives
            if hard_logits is not None:
                denom_logits = torch.cat([scaled_logits, hard_neg_weight * hard_logits], dim=1)  # [N, M+K]
                loss = F.cross_entropy(denom_logits, target, reduction=reduction)
            else:
                loss = F.cross_entropy(scaled_logits, target, reduction=reduction)

        return loss


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, use_symmetric_loss: bool = False):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(use_symmetric_loss)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        hard_neg: Tensor | None = None,   # shape [N, K, D] (2-D mode) or [B, K, D] (late interaction)
        x_task_ids: Tensor = None,
        y_task_ids: Tensor = None,
        target: Tensor = None,
        temperature: float = 0.02,
        inter_task_temperature: float | None = None,
        hard_neg_weight: float = 1.0,
        **kwargs
    ):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_x_task_ids = self.gather_tensor(x_task_ids)
        dist_y_task_ids = self.gather_tensor(y_task_ids)

        dist_hard_neg = None
        if hard_neg is not None:
            dist_hard_neg = self.gather_tensor(hard_neg)  # handles 2-D or 3-D tensors generically

        loss = super().__call__(
            x=dist_x,
            y=dist_y,
            x_task_ids=dist_x_task_ids,
            y_task_ids=dist_y_task_ids,
            target=target,
            temperature=temperature,
            inter_task_temperature=inter_task_temperature,
            hard_neg=dist_hard_neg,
            hard_neg_weight=hard_neg_weight,
            **kwargs
        )
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t: Tensor):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)

class InExampleContrastiveLoss:
    """
    Categorization loss: cross_entropy of 1 out of K classes (target labels)
    x.shape=[bsz, hdim], y.shape=[bsz, num_label, hdim]
    """

    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: int = None, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature
        self.ndim = ndim

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean'):
        # print("gather InExampleContrastiveLoss")
        if torch.distributed.is_initialized():
            x = dist_utils.dist_gather(x)
            y = dist_utils.dist_gather(y)
        bsz, ndim = x.size(0), x.size(1)
        target = torch.zeros(bsz, dtype=torch.long, device=x.device)
        if self.ndim:
            ndim = self.ndim
            x = x[:, :ndim]
            y = y[:, :ndim]
        logits = torch.einsum('bod,bsd->bs', x.view(bsz, 1, ndim), y.view(bsz, -1, ndim)) * self.temperature
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, target, reduction=reduction)
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail
