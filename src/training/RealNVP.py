import torch
import torch.nn as nn
import math
import lightning as L
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    def __init__(self, dim, cond_dim=0):
        super(MaskGenerator, self).__init__()
        self.mask_cond = cond_dim > 0
        if self.mask_cond:
            self.pred_layer = nn.Linear(cond_dim, dim)
            nn.init.zeros_(self.pred_layer.bias)
        else:
            self.mask_param = nn.Parameter(2 * torch.rand(dim) - 1, requires_grad=True)

    def forward(self, x, cond=None):
        if self.mask_cond:
            logits = self.pred_layer(cond)
        else:
            logits = self.mask_param
        mask = torch.round(torch.sigmoid(logits))
        return x * mask

class BatchNormFlow(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        self.momentum = momentum
        self.eps = eps

    def forward(self, x, cond=None, reverse=False):
        if not reverse:
            if self.training:
                self.batch_mean = x.mean(0)
                self.batch_var = (x - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (x - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var + self.eps)).sum(-1, keepdim=True)
        
        else:
            if self.training:
                mean = getattr(self, "batch_mean", self.running_mean)
                var  = getattr(self, "batch_var", self.running_var)
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (x - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var + self.eps)).sum(-1, keepdim=True)


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask, cond_dim, reverse=False):
        super(CouplingLayer, self).__init__()
        self.mask = mask
        self.reverse = reverse
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        total_dim = input_dim + cond_dim

        self.scale_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 512), nn.Tanh(),
            nn.Linear(512, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 512), nn.ReLU(),
            nn.Linear(512, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.apply(self.init)

    @staticmethod
    def init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self, x, cond, reverse=False):
        
        masked_inputs = self.mask(x, cond)
        inv_masked_input = self.mask(x, cond)

        masked_inputs = torch.cat([masked_inputs, cond], -1)
        log_s = self.scale_net(masked_inputs)  
        t = self.translate_net(masked_inputs) 
        if not reverse:
            s = torch.exp(log_s)
            return inv_masked_input + x * s + t, log_s.sum(-1, keepdim=True)
        else:
            s = torch.exp(-log_s)
            return inv_masked_input + (x - t) * s, -log_s.sum(-1, keepdim=True)


class RealNVP(nn.Module):
    def __init__(self, device, input_dim, cond_dim=512, num_blocks=5, hidden_dim=1024, mask_cond=False, reverse=False, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.device = device
        self.eps = eps

        self.layers = nn.ModuleList()
        masks = [] 
        for i in range(num_blocks):
            if mask_cond:
                masks.append(MaskGenerator(input_dim, cond_dim))
            else:
                masks.append(MaskGenerator(input_dim))
            mask = masks[-1]
            
            self.layers.append(CouplingLayer(input_dim, hidden_dim, mask, cond_dim, reverse))
            self.layers.append(BatchNormFlow(input_dim, eps=eps))

    def forward(self, x, cond, reverse=False, logdets=None):
        if logdets is None:
            logdets = torch.zeros(x.size(0), 1, device=x.device)

        iterable = reversed(self.layers) if reverse else self.layers
        for m in iterable:
            x, logdet = m(x, cond, reverse)
            logdets += logdet

        return x, logdets


    def log_prob(self, x, cond=None):
        u, log_jacob = self.forward(x, cond, reverse=False)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi + self.eps)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples, cond, noise=None, sigma=1.0):
        device = next(self.parameters()).device
        if noise is None:
            noise = torch.randn(num_samples, self.input_dim, device=device) * sigma

        cond = cond.to(device)
        samples, _ = self.forward(noise, cond, reverse=True)

        return samples


class RealNVPLightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = RealNVP(
            device="cuda" if torch.cuda.is_available() else "cpu",
            input_dim=config["input_dim"],
            cond_dim=config["cond_dim"],
            num_blocks=config["num_blocks"],
            hidden_dim=config["hidden_dim"],
            mask_cond=config.get('mask_conditioning', False),
            reverse=False,
            eps=config.get("eps", 1e-6),
        )


    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
        )
        return opt

    def training_step(self, batch, batch_idx):
        conds, gts = batch

        pred_params = self.model.sample(num_samples = gts.shape[0], cond = conds)
        loss = F.l1_loss(pred_params, gts)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        conds, gts = batch

        pred_params = self.model.sample(num_samples=gts.shape[0], cond=conds)
        loss = F.l1_loss(pred_params, gts)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        conds, gts = batch
        pred_params = self.model.sample(num_samples=gts.shape[0], cond=conds)
        loss = F.l1_loss(pred_params, gts)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss
