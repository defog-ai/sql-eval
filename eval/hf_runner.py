from eval.eval import compare_df, query_postgres_db, subset_df
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.pruning import prune_metadata_str

def prepare_questions_df(questions_file, num_questions):
    question_query_df = pd.read_csv(questions_file, nrows=num_questions)
    question_query_df["generated_query"] = ""
    question_query_df["reason"] = ""
    question_query_df["error_msg"] = ""
    question_query_df["correct"] = 0
    question_query_df["subset"] = 0
    question_query_df["error_query_gen"] = 0
    question_query_df["error_db_exec"] = 0
    question_query_df["timeout"] = 0
    # add custom metrics below:
    question_query_df["latency_seconds"] = 0.0  # latency of query generation in seconds
    question_query_df["tokens_used"] = 0  # number of tokens used in query generation

    question_query_df.reset_index(inplace=True, drop=True)
    return question_query_df

def generate_prompt(prompt_file, question, db_name):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    
    pruned_metadata_str = prune_metadata_str(question, db_name)
    prompt = prompt.format(user_question = question, table_metadata_string=pruned_metadata_str)
    return prompt

def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True)
    return tokenizer, model

def run_hf_eval(
    questions_file : str,
    prompt_file: str,
    num_questions: int = None,
    model_name : str = "defog/starcoder-finetune-v3",
):
    print("preparing questions...")
    # get questions
    df = prepare_questions_df(questions_file, num_questions)

    # create a prompt for each question
    df["prompt"] = df[['question', 'db_name']].apply(lambda row: generate_prompt(prompt_file, row['question'], row['db_name']), axis=1)

    # initialize tokenizer and model
    tokenizer, model = get_tokenizer_model(model_name)
    
    print("questions prepared\nnow generating predictions...")
    # generate predictions
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    inputs = tokenizer(df["prompt"].tolist(), return_tensors="pt", padding=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,
        num_beams=4,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
    )
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    df["prediction"] = predictions
    df['generated_query'] = df['prediction'].apply(lambda x: x.split("```sql")[-1].split(";")[0].strip())

    # from here, just do the usual eval stuff

    # export results to CSV before doing anything else
    df.to_csv("hf_pred.csv", index=False)