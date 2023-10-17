from eval.openai_runner import run_openai_eval
from eval.hf_runner import run_hf_eval
from eval.anthropic_runner import run_anthropic_eval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions_file", type=str, required=True)
    parser.add_argument("-n", "--num_questions", type=int, default=None)
    parser.add_argument("-g", "--model_type", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-a", "--adapter", type=str)
    parser.add_argument("-f", "--prompt_file", type=str, required=True)
    parser.add_argument("-d", "--use_private_data", action="store_true")
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.model_type == "oa":
        if args.model is None:
            args.model = "gpt-3.5-turbo-0613"
        run_openai_eval(args)
    elif args.model_type == "anthropic":
        if args.model is None:
            args.model = "claude-2"
        run_anthropic_eval(args)
    elif args.model_type == "hf":
        run_hf_eval(args)
    else:
        raise ValueError(
            f"Invalid model type: {args.model_type}. Model type must be one of: 'oa', 'hf'"
        )
