import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    """
    Focal loss for multi-class classification (1 label out of C possible classes).

    Args:
        alpha (float): Weighting factor for the loss (for balancing overall scale).
            You can set it to something like 0.25 if you want to downweight easy negatives,
            or keep it at 1.0 if you just want the focusing effect.
        gamma (float): The focusing parameter in focal loss.
        reduction (str): 'mean', 'sum', or 'none'.

    Shape:
        - logits: [batch_size, num_classes]
        - target: [batch_size], each value in [0..num_classes-1]
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (Tensor) [B, C], the raw outputs of your model
            targets: (Tensor) [B], long dtype with values in [0..C-1]
        Returns:
            A scalar if reduction != 'none', or a tensor of shape [B] if reduction=='none'.
        """

        # 1) Compute log-softmax: shape => [B, C]
        log_probs = F.log_softmax(logits, dim=1)

        # 2) NLL loss, but per-sample (no immediate reduction).
        #    shape => [B], each entry is -log(prob_of_correct_class).
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")

        # 3) Convert to p_t = prob of the correct class => exp(-CE)
        #    shape => [B]
        pt = torch.exp(-ce_loss)

        # 4) Focal term = (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # 5) Combine everything: alpha * focal_term * CE
        loss = self.alpha * focal_term * ce_loss  # shape => [B]

        # 6) Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            # 'none': return per-sample focal loss, shape => [B]
            return loss


if __name__ == "__main__":
    # Example usage
    logits = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]], dtype=torch.float32)
    targets = torch.tensor([1, 2], dtype=torch.long)

    criterion = MultiClassFocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
    loss = criterion(logits, targets)
    print("Focal Loss:", loss.item())