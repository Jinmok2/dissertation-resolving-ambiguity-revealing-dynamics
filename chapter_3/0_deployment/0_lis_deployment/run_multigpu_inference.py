
import os
import argparse
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def split_dataframe(df, num_shards):
    """Split a DataFrame into approximately equal shards."""
    return np.array_split(df, num_shards)


def run_inference_on_gpu(args):
    """
    Run inference on a single GPU.
    This function is called by one process.
    """
    (
        shard_id,
        shard_df,
        gpu_id,
        model_path,
        model_type,
        batch_size,
        max_length,
        output_dir
    ) = args

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print(f"[GPU {gpu_id}] Loading model from {model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_prob_match = []
    all_prob_nonmatch = []
    all_pred_labels = []

    n = len(shard_df)

    for start_idx in tqdm(
        range(0, n, batch_size),
        desc=f"GPU {gpu_id} shard {shard_id}",
        position=gpu_id
    ):
        batch_df = shard_df.iloc[start_idx:start_idx + batch_size]

        names1 = batch_df["OrgName_1"].astype(str).tolist()
        names2 = batch_df["OrgName_2"].astype(str).tolist()

        encoded = tokenizer(
            names1,
            names2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        prob_nonmatch = probs[:, 0].detach().cpu().numpy()
        prob_match = probs[:, 1].detach().cpu().numpy()
        pred_labels = np.argmax(probs.detach().cpu().numpy(), axis=1)

        all_prob_nonmatch.extend(prob_nonmatch)
        all_prob_match.extend(prob_match)
        all_pred_labels.extend(pred_labels)

    result_df = shard_df.copy()
    result_df["Model"] = model_type
    result_df["PredictedLabel"] = all_pred_labels
    result_df["Probability_Match"] = all_prob_match
    result_df["Probability_NonMatch"] = all_prob_nonmatch
    result_df["SimilarityScore"] = result_df["Probability_Match"]

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"deployment_pairwise_scores_shard_{shard_id}.csv"
    )

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[GPU {gpu_id}] Saved {output_path}", flush=True)

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate_pairs", type=str, default="deployment_candidate_pairs.csv")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="SciBERT")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="deployment_pairwise_score_shards")
    parser.add_argument("--combined_output", type=str, default="deployment_pairwise_scores.csv")
    parser.add_argument("--num_gpus", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.candidate_pairs):
        raise FileNotFoundError(f"Candidate pair file not found: {args.candidate_pairs}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    available_gpus = torch.cuda.device_count()

    if available_gpus == 0:
        raise RuntimeError("No CUDA GPU is available.")

    num_gpus = args.num_gpus or available_gpus
    num_gpus = min(num_gpus, available_gpus)

    print(f"Available GPUs: {available_gpus}", flush=True)
    print(f"Using GPUs: {num_gpus}", flush=True)

    print("Loading candidate pairs...", flush=True)
    candidate_pairs_df = pd.read_csv(args.candidate_pairs, encoding="utf-8-sig")

    print(f"Total candidate pairs: {len(candidate_pairs_df):,}", flush=True)

    shards = split_dataframe(candidate_pairs_df, num_gpus)

    worker_args = []

    for shard_id, shard_df in enumerate(shards):
        gpu_id = shard_id
        print(f"Shard {shard_id}: {len(shard_df):,} pairs -> GPU {gpu_id}", flush=True)

        worker_args.append(
            (
                shard_id,
                shard_df,
                gpu_id,
                args.model_path,
                args.model_type,
                args.batch_size,
                args.max_length,
                args.output_dir
            )
        )

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=num_gpus) as pool:
        output_paths = pool.map(run_inference_on_gpu, worker_args)

    print("All shards finished.", flush=True)
    print(output_paths, flush=True)

    print("Combining shard outputs...", flush=True)

    score_dfs = []

    for path in sorted(output_paths):
        temp_df = pd.read_csv(path, encoding="utf-8-sig")
        score_dfs.append(temp_df)

    combined_df = pd.concat(score_dfs, ignore_index=True)

    combined_df.to_csv(args.combined_output, index=False, encoding="utf-8-sig")

    print(f"Saved combined output: {args.combined_output}", flush=True)
    print(f"Number of scored pairs: {len(combined_df):,}", flush=True)


if __name__ == "__main__":
    main()
