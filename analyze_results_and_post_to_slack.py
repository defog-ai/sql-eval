import os
import argparse
import pandas as pd
from slack_sdk import WebClient
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_names", nargs="+", type=str, required=True)
    model_names = parser.parse_args().model_names

    # Load the results
    results = []
    for model_name in model_names:
        fnames = [i for i in os.listdir(f"results/{model_name}") if i.endswith(".csv")]
        for fname in fnames:
            checkpoint = fname.split(model_name)[1].split("_")[1]
            eval_type = fname.split("_")[-1]
            tdf = pd.read_csv(f"results/{model_name}/{fname}")
            tdf["checkpoint"] = checkpoint
            tdf["eval_type"] = eval_type
            tdf["model_name"] = model_name
            results.append(tdf)

    results = pd.concat(results)

    # create a graph of the average correct for each model, with each model as a line and each checkpoint as a point on the x axis
    for model_name in model_names:
        avg_correct = (
            results[
                (~results.query_category.isin(["cat_a", "cat_b", "cat_c"]))
                & (results.model_name == model_name)
            ]
            .groupby(["eval_type", "checkpoint"])["correct"]
            .mean()
            .unstack()
        )
        avg_correct = avg_correct.reset_index()
        avg_correct = avg_correct.melt(
            id_vars=["eval_type"], var_name="checkpoint", value_name="correct"
        )
        print(avg_correct)
        facet_plot = sns.relplot(
            data=avg_correct,
            x="checkpoint",
            y="correct",
            col="eval_type",
            kind="scatter",
            col_wrap=3,
            figsize=(8, 4.5),
        )
        # save the graph
        # this will get overwritten each time the script is run, but that's okay
        facet_plot.figure.savefig(f"results/avg_correct_{model_name}.png")

        # post the graph to slack
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        slack_client.files_upload_v2(
            channel="C07940SRVM5",  # id of the eval-results channel
            title=f"Average Correct for {model_name}",
            file=f"results/avg_correct_{model_name}.png",
            initial_comment="A set of evals just finished running! Here's the average correct rate for each model and each checkpoint that was in the evals (excluding idk questions).",
        )
