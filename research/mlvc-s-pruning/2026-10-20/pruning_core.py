"""Paired, physically compact gated-FFN pruning for the pinned MLVC-S model.

These are research implementations, not a reproduction of any published recipe.
Calibration-based rankings fail explicitly if their required evidence is absent.
"""
import copy
import math
import torch
from torch import nn


def gated_ffns(model, scope='decoder'):
    scopes = {
        'reconstruction': ('recon_generation_net.',),
        'main_decoder': ('feature_extractor.', 'decoder.', 'recon_generation_net.'),
        'decoder': ('feature_adaptor_i.', 'feature_adaptor_p.', 'feature_extractor.',
                    'hyper_decoder.', 'temporal_prior_encoder.', 'y_prior_fusion.',
                    'y_spatial_prior.', 'decoder.', 'recon_generation_net.'),
        'active_decoder': ('feature_adaptor_p.', 'feature_extractor.',
                           'hyper_decoder.', 'temporal_prior_encoder.', 'y_prior_fusion.',
                           'y_spatial_prior.', 'decoder.', 'recon_generation_net.'),
        'all': ('',),
    }
    for name, module in model.named_modules():
        if not name.startswith(scopes[scope]) or not hasattr(module, 'ffn'):
            continue
        f = module.ffn
        if len(f) != 3 or not isinstance(f[0], nn.Conv2d) or not isinstance(f[2], nn.Conv2d):
            raise ValueError(f'Unsupported FFN layout: {name}')
        mode = str(getattr(f[1], 'chunk_mode', '')).lower()
        if 'gated' not in mode or f[0].out_channels != f[2].in_channels * 2:
            raise ValueError(f'Expected paired gated FFN: {name}')
        if f[0].groups != 1 or f[2].groups != 1:
            raise ValueError(f'Unsupported grouped FFN: {name}')
        yield name, module


def channel_scores(block, method, calibration=None, seed=0):
    """Higher means keep. Calibration RMS and Taylor are per gate/value product."""
    first, last = block.ffn[0], block.ffn[2]
    width = last.in_channels
    w = first.weight.detach().float().flatten(1)
    gate, value = w.chunk(2, 0)
    outgoing = last.weight.detach().float().transpose(0, 1).flatten(1)
    if method in ('l1', 'l2'):
        p = 1 if method == 'l1' else 2
        return gate.norm(p=p, dim=1) * value.norm(p=p, dim=1) * outgoing.norm(p=p, dim=1)
    if method == 'fpgm':
        paired = torch.cat((gate, value, outgoing), dim=1)
        # Geometric-median redundancy score, computed in chunks to bound memory.
        return torch.cat([torch.cdist(x, paired).sum(1) for x in paired.split(64)])
    if method == 'random':
        g = torch.Generator(device='cpu').manual_seed(seed)
        return torch.rand(width, generator=g).to(w.device)
    if method in ('activation_rms', 'taylor'):
        if calibration is None or method not in calibration:
            raise ValueError(f'{method} requires independent calibration data')
        scores = torch.as_tensor(calibration[method], device=w.device, dtype=torch.float32)
        if scores.shape != (width,) or not torch.isfinite(scores).all():
            raise ValueError(f'Invalid calibration scores: {method}')
        return scores
    raise ValueError(f'Unknown ranking: {method}')


def select_channels(scores, remove_fraction, alignment=8):
    if not 0 <= remove_fraction < 1 or alignment < 1:
        raise ValueError('Invalid removal fraction or alignment')
    width = len(scores)
    if not torch.isfinite(scores).all():
        raise ValueError('Non-finite saliency')
    keep_n = width if remove_fraction == 0 else max(alignment, math.ceil(width * (1-remove_fraction) / alignment) * alignment)
    keep_n = min(width, keep_n)
    return torch.argsort(scores, descending=True, stable=True)[:keep_n].sort().values


def compact_ffn(block, keep):
    """Return a new block. Input/output widths, residuals and alpha remain intact."""
    out = copy.deepcopy(block)
    first, last = out.ffn[0], out.ffn[2]
    width = last.in_channels
    keep = torch.as_tensor(keep, device=first.weight.device, dtype=torch.long)
    if keep.numel() == 0 or keep.ndim != 1 or keep.unique().numel() != keep.numel():
        raise ValueError('Channel indices must be nonempty and unique')
    if keep.min() < 0 or keep.max() >= width:
        raise ValueError('Channel index out of range')
    paired = torch.cat((keep, keep + width))
    first.weight = nn.Parameter(first.weight.detach()[paired].clone(), requires_grad=first.weight.requires_grad)
    if first.bias is not None:
        first.bias = nn.Parameter(first.bias.detach()[paired].clone(), requires_grad=first.bias.requires_grad)
    first.out_channels = 2 * keep.numel()
    last.weight = nn.Parameter(last.weight.detach()[:, keep].clone(), requires_grad=last.weight.requires_grad)
    last.in_channels = keep.numel()
    return out


def prune_model(model, method, remove_fraction, scope='decoder', calibration=None, seed=0):
    """Return a compact copy plus replayable layout; no source model mutation."""
    out = copy.deepcopy(model)
    layout = {}
    for idx, (name, block) in enumerate(list(gated_ffns(out, scope))):
        evidence = None if calibration is None else calibration.get(name)
        scores = channel_scores(block, method, evidence, seed + idx)
        keep = select_channels(scores, remove_fraction)
        parent_name, _, child_name = name.rpartition('.')
        parent = out.get_submodule(parent_name) if parent_name else out
        parent._modules[child_name] = compact_ffn(block, keep)
        layout[name] = {'original_width': block.ffn[2].in_channels, 'keep': keep.cpu().tolist()}
    if not layout:
        raise ValueError(f'No gated FFNs in scope {scope}')
    return out, layout


class ChannelCalibration:
    """Collect activation RMS and |activation * RD-gradient| on calibration only."""
    def __init__(self, model, scope='decoder'):
        self.sums, self.counts, self.grad_counts, self.handles = {}, {}, {}, []
        for name, block in gated_ffns(model, scope):
            self.sums[name] = {}
            self.counts[name] = 0
            self.grad_counts[name] = 0
            self.handles.append(block.ffn[1].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def hook(module, inputs, z):
            axes = (0, 2, 3)
            n = z.shape[0] * z.shape[2] * z.shape[3]
            self.counts[name] += n
            rms = z.detach().float().square().sum(axes).cpu()
            old = self.sums[name].get('square')
            self.sums[name]['square'] = rms if old is None else old + rms
            if z.requires_grad:
                value = z.detach()
                def gradient(g):
                    t = (value.float() * g.detach().float()).abs().sum(axes).cpu()
                    old_t = self.sums[name].get('taylor')
                    self.sums[name]['taylor'] = t if old_t is None else old_t + t
                    self.grad_counts[name] += n
                z.register_hook(gradient)
        return hook

    def result(self):
        out = {}
        for name, sums in self.sums.items():
            if not self.counts[name]:
                continue
            out[name] = {'activation_rms': (sums['square']/self.counts[name]).sqrt().tolist()}
            if self.grad_counts[name]:
                out[name]['taylor'] = (sums['taylor']/self.grad_counts[name]).tolist()
        return out

    def close(self):
        for h in self.handles:
            h.remove()
