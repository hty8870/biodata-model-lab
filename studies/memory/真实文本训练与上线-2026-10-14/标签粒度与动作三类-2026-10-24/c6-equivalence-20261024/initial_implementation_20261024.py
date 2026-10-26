from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import action_common_20261024 as AC
from research.memory_relations import finetune as FT

SEED = 20261024
PARENTS = (0, 0, 0, 0, 1, 2)


class TinyClassifier(nn.Module):
    def __init__(self, labels, dtype):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(16, 384), nn.Tanh())
        self.classifier = nn.Linear(384, labels)
        self.to(dtype=dtype)

    def forward(self, values):
        return self.classifier(self.backbone(values))


def pair(dtype, perturb=False):
    torch.manual_seed(SEED)
    a = TinyClassifier(3, dtype)
    nn.init.normal_(a.classifier.weight, std=.02)
    nn.init.zeros_(a.classifier.bias)
    c = TinyClassifier(6, dtype)
    c.backbone.load_state_dict(a.backbone.state_dict())
    with torch.no_grad():
        for row, parent in enumerate(PARENTS):
            c.classifier.weight[row].copy_(a.classifier.weight[parent])
            c.classifier.bias[row].copy_(a.classifier.bias[parent] - (math.log(4) if parent == 0 else 0))
        if perturb:
            noise = torch.randn_like(c.classifier.weight[:4]) * .01
            noise -= noise.mean(dim=0)
            c.classifier.weight[:4].add_(noise)
    return a, c


def aggregate(logits):
    if logits.shape[1] == 3:
        return logits if logits.dtype == torch.float64 else AC.coarse_logits(logits, "a3")
    if logits.dtype != torch.float64:
        return AC.coarse_logits(logits, "f6")
    return torch.stack((logits[:, :4].logsumexp(dim=1), logits[:, 4], logits[:, 5]), dim=1)


def max_abs(values):
    return float(values.detach().abs().max())


def spread(head, gradients=False):
    weight = head.weight.grad if gradients else head.weight
    bias = head.bias.grad if gradients else head.bias
    return max(max_abs(weight[:4] - weight[0]), max_abs(bias[:4] - bias[0]))


def grad_comparison(a, c):
    return {
        "encoder_grad_error": max(max_abs(pa.grad - pc.grad) for pa, pc in zip(a.backbone.parameters(), c.backbone.parameters())),
        "coexist_grad_times4_error": max(max_abs(a.classifier.weight.grad[0] - c.classifier.weight.grad[0] * 4),
                                          max_abs(a.classifier.bias.grad[0] - c.classifier.bias.grad[0] * 4)),
        "other_head_grad_error": max(max_abs(a.classifier.weight.grad[1:] - c.classifier.weight.grad[4:]),
                                      max_abs(a.classifier.bias.grad[1:] - c.classifier.bias.grad[4:])),
        "copy_gradient_spread": spread(c.classifier, gradients=True),
    }


def norm32(model):
    return sum(float(p.grad.detach().float().square().sum()) for p in model.parameters() if p.grad is not None) ** .5


def clip(model, maximum):
    before = norm32(model)
    if maximum is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), maximum, error_if_nonfinite=True)
        after = norm32(model)
        if after > maximum:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(maximum / (after + 1e-6))
    return before, norm32(model)


def snapshot(a, c, probe):
    with torch.no_grad():
        a_logits = aggregate(a(probe))
        c_logits = aggregate(c(probe))
        delta = (a_logits.softmax(dim=-1) - c_logits.softmax(dim=-1)).abs()
        return {
            "probability_error": float(delta.max()),
            "coarse_logit_error": max_abs(a_logits - c_logits),
            "encoder_parameter_error": max(max_abs(pa - pc) for pa, pc in zip(a.backbone.parameters(), c.backbone.parameters())),
            "head_weight_error": max_abs(a.classifier.weight - c.classifier.weight[[0, 4, 5]]),
            "copy_parameter_spread": spread(c.classifier),
            "probe_argmax_disagreements": int((a_logits.argmax(dim=1) != c_logits.argmax(dim=1)).sum()),
        }


def identities():
    a, c = pair(torch.float64)
    generator = torch.Generator().manual_seed(SEED + 1)
    x = torch.randn(12, 16, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(12) % 3
    fine_labels = torch.tensor([int(i // 3) % 4 if i % 3 == 0 else (4 if i % 3 == 1 else 5) for i in range(12)])
    a_logits, c_logits = a(x), c(x)
    a_loss = F.cross_entropy(a_logits, labels)
    c_loss = F.cross_entropy(aggregate(c_logits), labels)
    x_gradient = torch.autograd.grad(a_loss, x, retain_graph=True)[0]
    direction = torch.randn(12, 16, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    epsilon = 1e-5
    with torch.no_grad():
        fd = (F.cross_entropy(a(x + epsilon * direction), labels) - F.cross_entropy(a(x - epsilon * direction), labels)) / (2 * epsilon)
    directional_error = abs(float(fd) - float((x_gradient * direction).sum()))
    a_loss.backward()
    c_loss.backward()
    gradients = grad_comparison(a, c)
    a_norm2 = sum(float(p.grad.square().sum()) for p in a.parameters())
    c_norm2 = sum(float(p.grad.square().sum()) for p in c.parameters())
    coexist_norm2 = float(a.classifier.weight.grad[0].square().sum() + a.classifier.bias.grad[0].square())
    norm_identity_error = abs(a_norm2 - c_norm2 - .75 * coexist_norm2)
    coarse_error = max_abs(a_logits - aggregate(c_logits))
    loss_error = abs(float(a_loss.detach() - c_loss.detach()))
    raw_head_error = max_abs(a.classifier.weight.grad[0] - c.classifier.weight.grad[0])
    fine_logits = c_logits.detach().float()
    fine_prob = fine_logits.softmax(dim=1)
    direct = torch.stack((fine_prob[:, :4].sum(dim=1), fine_prob[:, 4], fine_prob[:, 5]), dim=1)
    public_error = max_abs(direct - AC.coarse_logits(fine_logits, "f6").softmax(dim=1))
    with torch.no_grad():
        broken = c_logits.detach().clone()
        broken[:, :4] += math.log(4)
        missing_offset_error = max_abs(a_logits.softmax(dim=1) - aggregate(broken).softmax(dim=1))
    c.zero_grad(set_to_none=True)
    fine_loss = F.cross_entropy(c(x.detach()), fine_labels)
    fine_loss.backward()
    fine_gradient_spread = spread(c.classifier, gradients=True)
    checks = {
        "initial_logits_equal": coarse_error < 1e-12,
        "initial_loss_equal": loss_error < 1e-12,
        "encoder_gradient_equal": gradients["encoder_grad_error"] < 1e-12,
        "coexist_gradient_quarter": gradients["coexist_grad_times4_error"] < 1e-12,
        "other_head_gradient_equal": gradients["other_head_grad_error"] < 1e-12,
        "copy_gradient_symmetric": gradients["copy_gradient_spread"] < 1e-12,
        "norm_squared_decomposition": norm_identity_error < 1e-12,
        "finite_difference_matches_autograd": directional_error < 1e-9,
        "independent_probability_sum_matches_public_helper": public_error < 2e-7,
        "negative_missing_log4_detected": missing_offset_error > .1,
        "negative_raw_gradients_not_equal": raw_head_error > 1e-3,
        "fine_supervision_breaks_gradient_symmetry": fine_gradient_spread > 1e-3,
    }
    assert all(checks.values()), checks
    return {"checks": checks, "coarse_logit_error": coarse_error, "loss_error": loss_error,
            **gradients, "norm_squared_identity_error": norm_identity_error,
            "finite_difference_error": directional_error, "public_helper_error": public_error,
            "missing_log4_probability_error": missing_offset_error, "raw_head_gradient_difference": raw_head_error,
            "fine_gradient_spread": fine_gradient_spread, "a3_initial_gradient_norm": a_norm2 ** .5,
            "c6_initial_gradient_norm": c_norm2 ** .5}


def optimizer(model, kind, epsilon):
    if kind == "sgd":
        return torch.optim.SGD([{"params": model.backbone.parameters(), "lr": FT.LR_BACKBONE},
                                {"params": model.classifier.parameters(), "lr": FT.LR_HEAD}])
    result = FT.build_joint_optimizer(model, train_backbone=True)
    for group in result.param_groups:
        group["eps"] = epsilon
    return result


def trajectory(name, *, dtype=torch.float64, kind="adamw", epsilon=1e-8, maximum=None,
               correct_sgd=False, bf16=False, perturb=False):
    a, c = pair(dtype, perturb=perturb)
    oa, oc = optimizer(a, kind, epsilon), optimizer(c, kind, epsilon)
    generator = torch.Generator().manual_seed(SEED + 2)
    batches = torch.randn(60, 12, 16, generator=generator, dtype=torch.float64).to(dtype)
    probe = torch.randn(48, 16, generator=generator, dtype=torch.float64).to(dtype)
    initial = snapshot(a, c, probe)
    records = []
    first_grad = None
    max_gradient_spread = 0.0
    for step, batch in enumerate(batches, 1):
        labels = (torch.arange(12) + step) % 3
        losses = []
        for model, opt in ((a, oa), (c, oc)):
            opt.zero_grad(set_to_none=True)
            context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
            with context:
                logits = aggregate(model(batch))
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            losses.append(float(loss.detach()))
        if step == 1:
            first_grad = grad_comparison(a, c)
        max_gradient_spread = max(max_gradient_spread, spread(c.classifier, gradients=True))
        if correct_sgd:
            c.classifier.weight.grad[:4].mul_(4)
            c.classifier.bias.grad[:4].mul_(4)
        na, ca = clip(a, maximum)
        nc, cc = clip(c, maximum)
        oa.step()
        oc.step()
        assert all(torch.isfinite(p).all() for model in (a, c) for p in model.parameters())
        records.append({"case": name, "step": step, **snapshot(a, c, probe),
                        "a3_loss": losses[0], "c6_loss": losses[1], "loss_error": abs(losses[0] - losses[1]),
                        "a3_gradient_norm": na, "c6_gradient_norm": nc,
                        "a3_clipped_norm": ca, "c6_clipped_norm": cc})
    maximum_fields = ("probability_error", "coarse_logit_error", "encoder_parameter_error", "head_weight_error",
                      "copy_parameter_spread", "loss_error", "probe_argmax_disagreements")
    summary = {"name": name, "dtype": str(dtype), "optimizer": kind, "epsilon": epsilon,
               "clip_norm": maximum, "sgd_child_gradient_multiplier": 4 if correct_sgd else 1,
               "training_autocast": "cpu_bfloat16" if bf16 else "none", "asymmetric_init_control": perturb,
               "updates_per_model": len(records), "initial": initial, "first_gradient_comparison": first_grad,
               "first_update": records[0], "last_update": records[-1],
               "max_over_updates": {field: max(row[field] for row in records) for field in maximum_fields},
               "copy_gradient_max_spread": max_gradient_spread,
               "clipped_updates": {"a3": sum(row["a3_gradient_norm"] > maximum for row in records) if maximum is not None else 0,
                                   "c6": sum(row["c6_gradient_norm"] > maximum for row in records) if maximum is not None else 0},
               "evaluation_probe_n": len(probe), "head_width": 384, "dropout": 0,
               "lr_backbone": FT.LR_BACKBONE, "lr_head": FT.LR_HEAD,
               "weight_decay": FT.WEIGHT_DECAY if kind == "adamw" else 0.0}
    return summary, records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = {"action_common": ROOT.parent / "action_common_20261024.py",
               "finetune": args.repo_root / "research/memory_relations/finetune.py",
               "original_design_lock": ROOT.parent / "design-lock.json",
               "original_evaluation_lock": ROOT.parent / "evaluation-lock.json",
               "confirmation_lock": ROOT.parent / "confirmation-v2/lock.json"}
    before = {name: sha(path) for name, path in sources.items()}
    started = time.perf_counter()
    identity = identities()
    specifications = [
        ("sgd_same_lr", {"kind": "sgd"}),
        ("sgd_corrected_child_x4", {"kind": "sgd", "correct_sgd": True}),
        ("adamw_ideal_eps0", {"epsilon": 0.0}),
        ("adamw_actual_eps", {}),
        ("adamw_epsilon_stress", {"epsilon": 1e-3}),
        ("adamw_actual_clip1", {"maximum": 1.0}),
        ("adamw_float32_clip1", {"dtype": torch.float32, "maximum": 1.0}),
        ("adamw_cpu_bf16_clip1", {"dtype": torch.float32, "maximum": 1.0, "bf16": True}),
        ("adamw_asymmetric_control", {"epsilon": 0.0, "perturb": True}),
    ]
    cases, all_records = {}, []
    for name, options in specifications:
        summary, records = trajectory(name, **options)
        cases[name] = summary
        all_records.extend(records)
        print(json.dumps({"case": name, "max_probability_error": summary["max_over_updates"]["probability_error"],
                          "copy_spread": summary["max_over_updates"]["copy_parameter_spread"],
                          "clip_counts": summary["clipped_updates"]}), flush=True)
    checks = {
        "ordinary_sgd_detects_different_trajectory": cases["sgd_same_lr"]["max_over_updates"]["probability_error"] > 1e-6,
        "corrected_sgd_matches": cases["sgd_corrected_child_x4"]["max_over_updates"]["probability_error"] < 1e-11,
        "ideal_adamw_matches": cases["adamw_ideal_eps0"]["max_over_updates"]["probability_error"] < 1e-10,
        "epsilon_stress_detects_difference": cases["adamw_epsilon_stress"]["max_over_updates"]["probability_error"] > 1e-5,
        "clip_condition_exercised": all(cases["adamw_actual_clip1"]["clipped_updates"][key] > 0 for key in ("a3", "c6")),
        "asymmetric_init_detected": cases["adamw_asymmetric_control"]["initial"]["probability_error"] > 1e-5,
        "symmetric_copies_remain_tied": all(case["max_over_updates"]["copy_parameter_spread"] < 1e-12
                                            and case["copy_gradient_max_spread"] < 1e-12
                                            for name, case in cases.items() if name != "adamw_asymmetric_control"),
        "source_hashes_unchanged": before == {name: sha(path) for name, path in sources.items()},
    }
    assert all(checks.values()), checks
    result = {"status": "passed", "seed": SEED, "identity": identity, "trajectory_checks": checks, "cases": cases,
              "source_sha256": before, "script_sha256": sha(__file__),
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                              "threads": torch.get_num_threads(), "deterministic_algorithms": True},
              "total_tiny_optimizer_updates": len(all_records) * 2, "seconds": time.perf_counter() - started,
              "scope": "algebra and synthetic tensor diagnostic; not transformer performance evaluation",
              "transformer_training_run": False, "weights_or_optimizer_saved": False, "gpu_used": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    steps = args.output.with_suffix(".steps.jsonl")
    steps.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in all_records), encoding="utf-8")
    print(json.dumps({"status": result["status"], "seconds": result["seconds"], "identity_checks": len(identity["checks"]),
                      "trajectory_checks": len(checks), "tiny_updates": result["total_tiny_optimizer_updates"]}), flush=True)


if __name__ == "__main__":
    main()
