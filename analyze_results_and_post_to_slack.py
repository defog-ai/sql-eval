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
            eval_type = fname.split("_")[-1].replace(".csv", "")
            # get cot_type based on whether there is _cot in the filename
            if "_cot" in fname:
                cot_inference = "cot"
            else:
                cot_inference = "no_cot"
            tdf = pd.read_csv(f"results/{model_name}/{fname}")
            tdf["model"] = model_name
            tdf["checkpoint"] = checkpoint
            tdf["eval_type"] = eval_type
            tdf["cot_inference"] = cot_inference
            tdf["model_name"] = model_name
            results.append(tdf)

    results = pd.concat(results)

    # create a graph of the average correct for each model, with each model as a line and each checkpoint as a point on the x axis
    avg_correct = (
        results.groupby(["model", "eval_type", "checkpoint", "cot_inference"])[
            "correct"
        ]
        .mean()
        .reset_index()
    )
    avg_correct = avg_correct.melt(
        id_vars=["model", "eval_type", "checkpoint", "cot_inference"],
        var_name="metric",
        value_name="correct_pct",
    )
    # arrange order of eval_type to be basic, v1, advanced, idk
    avg_correct["eval_type"] = pd.Categorical(
        avg_correct["eval_type"],
        categories=["basic", "v1", "advanced", "idk"],
        ordered=True,
    )
    print(avg_correct.drop(columns=["metric"]))
    facet_plot = sns.relplot(
        data=avg_correct,
        x="checkpoint",
        y="correct_pct",
        hue="model",
        style="cot_inference",
        col="eval_type",
        kind="line",
        col_wrap=3,
    )
    # add grid lines to all subplots
    for ax in facet_plot.axes:
        ax.grid(True, linestyle="--")

    plt.show()
    # save the graph
    # this will get overwritten each time the script is run, but that's okay
    facet_plot.figure.savefig(f"results/avg_correct_{model_name}.png")
    fnames = sorted(
        [i for i in os.listdir(f"results/{model_name}") if i.endswith(".csv")]
    )
    fnames = "\n".join([i.replace(".csv", "") for i in fnames])

    # post the graph to slack
    slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    slack_client.files_upload_v2(
        channel="C07940SRVM5",  # id of the eval-results channel
        title=f"Average Correct for {model_name}",
        file=f"results/avg_correct_{model_name}.png",
        initial_comment=f"""A set of evals just finished running for model `{model_name}`! The graph below has the average correct rate for each model and each checkpoint that was in the evals (excluding idk questions).
Additionally, if you want to see the raw data for any run in eval-visualizer, you can paste one of the following run names into the Eval Visualizer search bar:

```
{fnames}
```
""",
    )
