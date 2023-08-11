from eval.openai_runner import run_openai_eval
from eval.hf_runner import run_hf_eval
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions_file", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-g", "--model_type", type=str, required=True)
    parser.add_argument("-f", "--prompt_file", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("-n", "--num_questions", type=int, default=None)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.model_type == "oa_chat":
        run_openai_eval(args)
    elif args.model_type == "hf":
        run_hf_eval(
            questions_file=args.questions_file,
            prompt_file=args.prompt_file,
            num_questions=args.num_questions,
            model_name=args.model,
            output_file=args.output_file,
        )
    else:
        raise ValueError(
            f"Invalid model type: {args.model_type}. Model type must be one of: 'openai', 'hf'"
        )
