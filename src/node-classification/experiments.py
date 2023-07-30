import torch
import torch.nn.functional as F


def run_exp(
    exp_no,
    device,
    out,
    pos_out,
    neg_out,
    pos_vec_dist,
    neg_vec_dist,
    POS_DIST,
    NEG_DIST,
    DEG,
):
    k0 = 2
    k1 = 2
    delta = 1

    if exp_no == 17:
        pos_loss = 0.5 * (
            F.logsigmoid((out * pos_out).sum(-1)).mean()
            - 0.15
            * (
                torch.log(torch.as_tensor(DEG, device=device))
                * torch.pow(
                    torch.log(
                        torch.div(
                            pos_vec_dist / k0,
                            0.05 + torch.as_tensor(POS_DIST, device=device),
                        )
                    ),
                    2,
                )
            ).mean()
        )
        neg_loss = 0.5 * (
            F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            - 0.15
            * (
                torch.log(torch.as_tensor(DEG, device=device))
                * torch.pow(
                    torch.log(
                        torch.div(
                            neg_vec_dist / k0,
                            0.05 + torch.as_tensor(NEG_DIST, device=device),
                        )
                    ),
                    2,
                )
            ).mean()
        )

    # CAFIN-N
    elif exp_no == 18:
        pos_loss = 0.5 * (F.logsigmoid((out * pos_out).sum(-1)).mean())
        neg_loss = 0.5 * (
            F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            - 0.05
            * (
                torch.log(torch.as_tensor(DEG, device=device))
                * torch.pow(
                    torch.log(
                        torch.div(
                            neg_vec_dist / k0,
                            0.05 + torch.as_tensor(NEG_DIST, device=device),
                        )
                    ),
                    2,
                )
            ).mean()
        )

    # CAFIN-P
    elif exp_no == 19:
        pos_loss = 0.5 * (
            F.logsigmoid((out * pos_out).sum(-1)).mean()
            - 0.05
            * (
                torch.log(torch.as_tensor(DEG, device=device))
                * torch.pow(
                    torch.log(
                        torch.div(
                            pos_vec_dist / k0,
                            0.05 + torch.as_tensor(POS_DIST, device=device),
                        )
                    ),
                    2,
                )
            ).mean()
        )
        neg_loss = 0.5 * (F.logsigmoid(-(out * neg_out).sum(-1)).mean())
    
    # Original
    else:
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()

    return pos_loss, neg_loss
