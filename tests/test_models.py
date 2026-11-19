import pytest
import torch

from geompm.config import ModelConfig
from geompm.geology.geo_loss import GeoLoss
from geompm.models import MODEL_NAMES, GeoMultiHeadAttention, ProductAffinity, build_model

B, C, P = 4, 6, 32


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, C, P, P, generator=g),
        torch.rand(B, 1, P, P, generator=g),
        torch.rand(B, 1, P, P, generator=g),
        torch.rand(B, generator=g),
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_returns_logits(name):
    model = build_model(ModelConfig(name=name, patch_size=P), C).eval()
    out = model(*batch())
    assert out.shape == (B,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_parameter_receives_gradient(name):
    """Regression test for the SHVCNN bug where forward() used a global ViT, so only the head trained."""
    model = build_model(ModelConfig(name=name, patch_size=P, affinity="learned"), C).train()
    loss_fn = GeoLoss(lam=0.1)
    x, w, gp, gc = batch()
    loss, _ = loss_fn(model(x, w, gp, gc), torch.randint(0, 2, (B,)).float(), w[:, 0, P // 2, P // 2])
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    # alpha of a geo-attention is legitimately unused when affinity is 'none'
    missing = [n for n in missing if not (name in ("vit", "cnn_vit") and n.endswith("alpha"))]
    assert not missing, missing
    assert loss_fn.raw_a.grad is not None and loss_fn.b.grad is not None


@pytest.mark.parametrize("name", ["shcnn", "geovit", "shvcnn"])
def test_constrained_models_depend_on_priors(name):
    model = build_model(ModelConfig(name=name, patch_size=P, dropout=0.0), C).eval()
    x, w, gp, gc = batch()
    base = model(x, w, gp, gc)
    assert not torch.allclose(base, model(x, torch.rand_like(w), gp, gc))  # w: conv modulation and/or attention
    assert not torch.allclose(base, model(x, w, gp, torch.zeros_like(gc)))  # g: classifier head


@pytest.mark.parametrize("name", ["cnn", "vit", "cnn_vit"])
def test_baselines_ignore_priors(name):
    model = build_model(ModelConfig(name=name, patch_size=P, dropout=0.0), C).eval()
    x, w, gp, gc = batch()
    a = model(x, w, gp, gc)
    b = model(x, torch.rand_like(w), torch.rand_like(gp), torch.rand_like(gc))
    assert torch.allclose(a, b)


def test_geo_attention_bias_changes_attention():
    """Regression test: the old M was a per-row constant, which softmax cancels."""
    torch.manual_seed(0)
    attn = GeoMultiHeadAttention(16, 4).eval()
    x = torch.randn(2, 9, 16)
    w_tok = torch.rand(2, 9)
    M = ProductAffinity()(w_tok)
    attn(x, M)
    with_geo = attn.last_attention
    attn(x, None)
    no_geo = attn.last_attention
    assert not torch.allclose(with_geo, no_geo)
    attn(x, torch.full_like(M, 3.7))  # the old, constant M: softmax cancels it exactly
    assert torch.allclose(attn.last_attention, no_geo, atol=1e-6)


def test_geoloss_lambda_zero_is_bce():
    logits, y, w = torch.randn(8), torch.randint(0, 2, (8,)).float(), torch.rand(8)
    loss, _ = GeoLoss(lam=0.0)(logits, y, w)
    assert torch.allclose(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, y))


def test_geoloss_penalises_disagreement_with_prior():
    loss_fn = GeoLoss(lam=1.0, learn_affine=False)
    y, w = torch.ones(4), torch.ones(4)
    agree, _ = loss_fn(torch.full((4,), 0.8), y, w)  # sigmoid(0.8) ~ sigmoid(a*1+0)
    disagree, parts = loss_fn(torch.full((4,), -3.0), y, w)
    assert parts["geo"] > 0 and disagree > agree
