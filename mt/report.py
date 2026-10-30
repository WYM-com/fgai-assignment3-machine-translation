"""Create tables and figures only from completed measured model outputs."""
import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import ROOT, digest, read_json, read_rows

NAMES = ["basic", "mt5", "llama", "gemma"]


def report():
    directory = ROOT / "results"
    missing = [n for n in NAMES if not (directory / f"{n}_metrics.json").exists()]
    if missing:
        raise RuntimeError("Cannot produce final comparison; missing measured results for " + ", ".join(missing))
    values = [read_json(directory / f"{n}_metrics.json") for n in NAMES]
    if len({r["test_hash"] for r in values}) != 1 or values[0]["test_hash"] != digest(read_rows("test")):
        raise ValueError("Models were not evaluated on the current identical test set")
    if len({r["config_hash"] for r in values}) != 1:
        raise ValueError("Configuration differs between runs")
    columns = ["model", "bleu", "chrf", "parameters", "training_seconds", "inference_seconds",
               "seconds_per_sentence", "device", "hardware", "dtype", "source_truncated_count", "output_limit_count"]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ["bleu", "chrf", "seconds_per_sentence"], ["BLEU (higher is better)", "chrF (higher is better)", "Seconds per sentence (lower is better)"]):
        ax.bar(NAMES, [r[key] for r in values], color=["#385170", "#507c90", "#da9351", "#9c637d"])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)
    winner = max(values, key=lambda r: r["bleu"])
    chrf_winner = max(values, key=lambda r: r["chrf"])
    lines = ["# Measured results", "", "All scores below come from the saved test predictions.", "",
        "| Model | BLEU | chrF | Parameters | Training seconds | Inference seconds |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in values:
        training = "N/A (prompting only)" if r["training_seconds"] is None else f'{r["training_seconds"]:.1f}'
        lines.append(f'| {r["model"]} | {r["bleu"]:.2f} | {r["chrf"]:.2f} | {r["parameters"]:,} | {training} | {r["inference_seconds"]:.2f} |')
    lines += ["", f'Best observed model by primary metric BLEU: **{winner["model"]}**, with {winner["bleu"]:.2f}. '
        f'The highest chrF belongs to **{chrf_winner["model"]}** ({chrf_winner["chrf"]:.2f}). '
        'These are descriptive rankings from one test sample, not statistical significance claims.', "",
        "The basic Transformer learns translation and its vocabulary only from the training split. mT5 starts from multilingual pretraining and receives supervised task adaptation. LLaMA and Gemma start from instruction-tuned checkpoints and receive an identical zero-shot instruction. Thus this experiment compares practical systems, not an isolated causal effect of architecture. Parameter count, pretraining exposure, tokenizer, precision and adaptation differ. Pretraining compute is excluded from the reported training time.", "",
        "Batch size is one during inference, decoding is greedy, and the maximum output budget is the same number of tokens. Token counts are not equivalent linguistic lengths across tokenizers. Encoder inputs are capped at the configured length; decoder-only prompts retain the full short source. Inspect source/output truncation counts before interpreting scores. Timings include tokenization and decoding after warm-up. Hardware and precision are recorded per model; cross-device speed rankings would be confounded.", "",
        "## Sampling temperature", ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, ax in zip(["llama", "gemma"], axes):
        experiments = read_json(directory / f"{name}_temperature_summary.json")
        sampled = [r for r in experiments if r["temperature"] != "greedy"]
        temps = sorted({r["temperature"] for r in sampled})
        means, stds = [], []
        for temp in temps:
            group = [r for r in sampled if r["temperature"] == temp]
            mean = statistics.mean(r["bleu"] for r in group)
            std = statistics.stdev(r["bleu"] for r in group) if len(group) > 1 else 0
            means.append(mean)
            stds.append(std)
            lines.append(f'- {name}, temperature {temp}: BLEU mean {mean:.2f}, sample SD {std:.2f}; chrF mean {statistics.mean(r["chrf"] for r in group):.2f} across {len(group)} seeds.')
        baseline = next(r for r in experiments if r["temperature"] == "greedy")
        lines.append(f'- {name}, greedy on the identical {baseline["count"]}-sentence subset: BLEU {baseline["bleu"]:.2f}, chrF {baseline["chrf"]:.2f}.')
        ax.errorbar(temps, means, yerr=stds, marker="o", capsize=4)
        ax.axhline(baseline["bleu"], linestyle="--", color="gray", label="Greedy baseline")
        ax.set(xlabel="Temperature", ylabel="BLEU", title=name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "temperature.png", dpi=180)
    plt.close(fig)
    lines += ["", "Higher temperature flattens the token distribution and can increase lexical variation or unsupported additions. Whether this helps overlap scores must be read from the measured values above; no monotonic effect is assumed. Top-p is fixed at 0.9 and top-k is disabled for all sampled runs. This exploratory test subset is not used to select a new decoding policy for the main comparison.", "", "## Translation review", "",
        "See `samples.csv` for aligned translations and `ethics_review.csv` for actual probe outputs. Review errors against the source, since valid paraphrases can have low reference overlap. Complete the human-review columns before submission; the probes are qualitative checks, not a validated safety benchmark."]
    (directory / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    predictions = {name: read_json(directory / f"{name}_predictions.json") for name in NAMES}
    with (directory / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", "reference"] + NAMES)
        for i, row in enumerate(predictions["basic"]):
            assert all(predictions[n][i]["id"] == row["id"] for n in NAMES)
            writer.writerow([row["id"], row["source"], row["reference"]] + [predictions[n][i]["prediction"] for n in NAMES])
    review_path = directory / "ethics_review.csv"
    # Preserve any existing human annotations on subsequent report runs.
    if not review_path.exists():
        with review_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "id", "concern", "source", "prediction", "review_criterion", "human_verdict", "explanation"])
            for name in NAMES:
                for row in read_json(directory / f"{name}_ethics.json"):
                    writer.writerow([name, row["id"], row["concern"], row["source"], row["prediction"], row["review"], "", ""])
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    before, tail = text.split(start, 1)
    _, after = tail.split(end, 1)
    updated = before + start + "\n" + "\n".join(lines) + "\n" + end + after
    status_start, status_end = "<!-- STATUS_START -->", "<!-- STATUS_END -->"
    before_status, remaining = updated.split(status_start, 1)
    _, after_status = remaining.split(status_end, 1)
    updated = before_status + status_start + "\nAll four model evaluations and both temperature studies are complete. " \
        "Measured results appear below. Review the actual translations and complete the ethical-review " \
        "worksheet before final submission.\n" + status_end + after_status
    readme.write_text(updated, encoding="utf-8")
    print("Generated measured comparison, plots, aligned translations, and review worksheet.")


if __name__ == "__main__":
    report()
