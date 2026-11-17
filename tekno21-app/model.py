import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

torch.backends.mha.set_fastpath_enabled(False)   # necesario para capturar la atención


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes=3, embed_dim=256, depth=4, heads=8,
                 mlp_ratio=4, dropout=0.1, pretrained=False):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool,
                                 r.layer1, r.layer2, r.layer3)
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)
        n_tokens = 14 * 14
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * mlp_ratio, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        f = self.proj(self.cnn(x))
        t = f.flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1) + self.pos
        t = self.encoder(t)
        return self.head(self.norm(t[:, 0]))


def _normalize(cam):
    mn = cam.amin(dim=(1, 2), keepdim=True)
    mx = cam.amax(dim=(1, 2), keepdim=True)
    return (cam - mn) / (mx - mn + 1e-8)


class AttentionRollout:
    def __init__(self, model, img_size=224):
        self.model, self.img_size, self.maps = model, img_size, []
        for layer in model.encoder.layers:
            mha = layer.self_attn
            if not hasattr(mha, "_orig_forward"):
                mha._orig_forward = mha.forward

            def fwd(*args, _orig=mha._orig_forward, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = True
                out, w = _orig(*args, **kwargs)
                self.maps.append(w.detach())
                return out, w

            mha.forward = fwd

    @torch.no_grad()
    def __call__(self, x):
        self.maps.clear()
        logits = self.model(x)
        n = self.maps[0].size(-1)
        eye = torch.eye(n, device=x.device)
        rollout = eye.expand(x.size(0), n, n)
        for A in self.maps:
            A = A + eye
            A = A / A.sum(dim=-1, keepdim=True)
            rollout = A @ rollout
        cls_att = rollout[:, 0, 1:]
        side = int(cls_att.size(-1) ** 0.5)
        cam = cls_att.reshape(-1, 1, side, side)
        cam = F.interpolate(cam, size=(self.img_size,) * 2, mode="bilinear",
                            align_corners=False).squeeze(1)
        return _normalize(cam).cpu().numpy(), logits.softmax(1).cpu().numpy()


class GradCAM:
    def __init__(self, model, target_layer, img_size=224):
        self.model, self.img_size = model, img_size
        self.acts, self.grads = None, None
        target_layer.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inp, out):
        self.acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        self.model.zero_grad()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            if target is None:
                target = logits.argmax(1)
            logits.gather(1, target.view(-1, 1)).sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.img_size,) * 2, mode="bilinear",
                            align_corners=False).squeeze(1)
        return _normalize(cam).detach().cpu().numpy(), logits.softmax(1).detach().cpu().numpy()