import os
import argparse
import pandas as pd
from slack_sdk import WebClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_names", nargs="+", type=str, required=True)
    model_names = parser.parse_args().model_names

    # Load the results
    results = []
    for model_name in model_names:
        fnames = [i for i in os.listdir(f"results/{model_name}") if i.endswith(".csv")]
        for fname in fnames:
            checkpoint = fname.split("_")[1]
            eval_type = fname.split("_")[-1]
            tdf = pd.read_csv(f"results/{model_name}/{fname}")
            tdf["checkpoint"] = checkpoint
            tdf["eval_type"] = eval_type
            tdf["model_name"] = model_name
            results.append(tdf)

    results = pd.concat(results)

    # first, get the average correct for each model, so that the index is the model name and each checkpoint is a column
    avg_correct = (
        results.groupby(["model_name", "checkpoint"])["correct"].mean().unstack()
    )
    avg_correct = avg_correct.reset_index()
    avg_correct = avg_correct.melt(
        id_vars=["model_name"], var_name="checkpoint", value_name="avg_correct"
    )
    print(avg_correct)
