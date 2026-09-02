#!/usr/bin/env python3
"""Generate the README figures from the canonical dissertation summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/dissertation/summary.json"
OUTPUT = ROOT / "assets/readme"
COLORS = {"full": "#334155", "k12": "#0f766e", "accent": "#d97706", "bad": "#b91c1c"}


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def finish(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def schedule_figure(data: dict) -> None:
    schedule = data["method"]["schedule"]
    xs = [0]
    ys = []
    for segment in schedule:
        ys.append(segment["visual_tokens"])
        xs.append(xs[-1] + segment["layers"])
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    for index, value in enumerate(ys):
        ax.hlines(value, xs[index], xs[index + 1], color=COLORS["k12"], linewidth=7)
        if index + 1 < len(ys):
            ax.vlines(xs[index + 1], ys[index + 1], value, color=COLORS["k12"], linewidth=2)
    ax.set(xlim=(0, 32), ylim=(-30, 620), xlabel="Decoder layer", ylabel="Visual tokens")
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_title("K12-QueryTopK-Z0 token schedule", loc="left", weight="bold")
    ax.grid(axis="y", alpha=0.2)
    finish(fig, "k12_schedule.png")


def quality_work_figure(data: dict) -> None:
    validation = data["refcoco"]["validation"]
    methods = ["Full", "K12-Z0"]
    work = [18432, data["method"]["visual_token_layer_work"]]
    acc = [100 * validation["full"]["acc_at_0_5"], 100 * validation["k12_z0"]["acc_at_0_5"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(work, acc, s=130, c=[COLORS["full"], COLORS["k12"]], zorder=3)
    for label, x, y in zip(methods, work, acc, strict=True):
        ax.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points", weight="bold")
    ax.set(xlabel="Visual token-layer work", ylabel="RefCOCO validation Acc@0.5 (%)")
    ax.set_title("Quality versus decoder token work", loc="left", weight="bold")
    ax.grid(alpha=0.2)
    finish(fig, "quality_vs_work.png")


def transfer_figure(data: dict) -> None:
    transfer = data["transfer"]
    labels = ["TextVQA", "POPE", "MME\n(normalised)"]
    values = [
        transfer["textvqa"]["delta_pp"],
        transfer["pope"]["delta_pp"],
        transfer["mme"]["delta"] / 10,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [COLORS["bad"] if value < 0 else COLORS["k12"] for value in values]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{values[0]:+.2f} pp", f"{values[1]:+.3f} pp", "+31.00 pts"], padding=4)
    ax.set_ylabel("Change from matching Full control")
    ax.set_title("K12-Z0 transfer is task-dependent", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "cross_benchmark_transfer.png")


def textvqa_figure(data: dict) -> None:
    textvqa = data["transfer"]["textvqa"]
    labels = ["Full", "K12-NoTail", "K12-Z0"]
    values = [
        100 * textvqa["full_soft_accuracy"],
        100 * textvqa["k12_no_tail_soft_accuracy"],
        100 * textvqa["k12_z0_soft_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=[COLORS["full"], COLORS["accent"], COLORS["k12"]], width=0.62)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
    ax.set(ylim=(0, 55), ylabel="TextVQA soft accuracy (%)")
    ax.set_title("Removing the zero tail recovers only 2.44 points", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "textvqa_no_tail.png")


def main() -> int:
    data = load_summary()
    schedule_figure(data)
    quality_work_figure(data)
    transfer_figure(data)
    textvqa_figure(data)
    print(f"Generated 4 README figures in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
