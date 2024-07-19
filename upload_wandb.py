import os
import pandas as pd
import wandb

# Step 1. Specify the folder where the result csv files are stored
result_folder = (
    "results/sqlcoder_8b_fullft_ds_012_llama3_join_after_mgn1_b1_0900_b2_0990_steps_600"
)
# Step 2. Specify the wandb run id
wandb_run_id = "fqianfsq"

# rest of the script logic
csv_files = []
for f in os.listdir(result_folder):
    if f.endswith(".csv"):
        csv_files.append(f)
print(f"Found {len(csv_files)} csv files in {result_folder}")

# Load results from csv file into dataframe
results_dfs = []
for csv_file_name in csv_files:
    file_path = os.path.join(result_folder, csv_file_name)
    df_i = pd.read_csv(file_path, comment="#")
    df_i["model"] = csv_file_name.rsplit(".csv", 1)[0]
    results_dfs.append(df_i)
results_df = pd.concat(results_dfs, ignore_index=True)
print(f"Loaded {results_df.shape[0]} results from {len(csv_files)} csv files")

s = results_df.groupby("model")["correct"].mean()
s = pd.DataFrame(s)
s["file_name"] = s.index
s["benchmark"] = s["file_name"].str.extract(r"_(advanced|basic|v1|idk)")
s["checkpoint"] = s["file_name"].str.extract(r"c(\d+)_").astype(int)
s["cot"] = s["file_name"].str.extract(r"_(cot)").fillna("no_cot")
s = s.reset_index(drop=True)

# Get unique checkpoints
checkpoints = s["checkpoint"].unique()
checkpoints.sort()
print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")

# Continue existing run, specifying the project and the run ID
run = wandb.init(project="huggingface", id=wandb_run_id, resume="must")

# get current step, so that we can log incrementally after it
# this is because wandb doesn't allow logging back to previous steps
current_step = run.step
print(f"Current step: {current_step}")

for checkpoint in checkpoints:
    checkpoint_metrics = {}
    for benchmark in ["advanced", "basic", "v1", "idk"]:
        for cot in ["cot", "no_cot"]:
            mask = (
                (s["checkpoint"] == checkpoint)
                & (s["benchmark"] == benchmark)
                & (s["cot"] == cot)
            )
            if mask.sum() == 1:
                row = s[mask]
                metric_name = f"vllm/{benchmark}"
                if cot == "cot":
                    metric_name += "_cot"
                metric_value = row["correct"].values[0]
                checkpoint_metrics[metric_name] = metric_value
    print(f"Logging checkpoint {checkpoint} metrics:")
    for k, v in checkpoint_metrics.items():
        print(f"\t{k}: {v}")
    # we log the metrics at the current step + checkpoint
    wandb.log(checkpoint_metrics, step=current_step + checkpoint)

# Finish the run
run.finish()