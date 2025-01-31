import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data-related parameters
    parser.add_argument("-q", "--questions_file", nargs="+", type=str, required=True)
    parser.add_argument("-n", "--num_questions", type=int, default=None)
    parser.add_argument("-db", "--db_type", type=str, required=True)
    parser.add_argument("-d", "--use_private_data", action="store_true")
    parser.add_argument("-dp", "--decimal_points", type=int, default=None)
    # model-related parameters
    parser.add_argument("-g", "--model_type", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-a", "--adapter", type=str)  # path to adapter
    parser.add_argument(
        "-an", "--adapter_name", type=str, default=None
    )  # only for use with production server
    parser.add_argument("--api_url", type=str)
    parser.add_argument("--api_type", type=str)
    # inference-technique-related parameters
    parser.add_argument("-f", "--prompt_file", nargs="+", type=str, required=True)
    parser.add_argument("-b", "--num_beams", type=int, default=1)
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=4
    )  # batch size, only relevant for the hf runner
    parser.add_argument("-c", "--num_columns", type=int, default=0)
    parser.add_argument("-s", "--shuffle_metadata", action="store_true")
    parser.add_argument("-k", "--k_shot", action="store_true")
    parser.add_argument(
        "--cot_table_alias", type=str, choices=["instruct", "pregen", ""], default=""
    )
    # execution-related parameters
    parser.add_argument("-o", "--output_file", nargs="+", type=str, required=True)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--logprobs", action="store_true")
    parser.add_argument("--upload_url", type=str)
    parser.add_argument("--run_name", type=str, required=False)
    parser.add_argument(
        "-qz", "--quantized", default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    # if questions_file is None, set it to the default questions file for the given db_type
    if args.questions_file is None:
        args.questions_file = f"data/questions_gen_{args.db_type}.csv"

    # check that questions_file matches db_type
    for questions_file in args.questions_file:
        if args.db_type not in questions_file and questions_file != "data/idk.csv":
            print(
                f"WARNING: Check that questions_file {questions_file} is compatible with db_type {args.db_type}"
            )

    if args.upload_url is None:
        args.upload_url = os.environ.get("SQL_EVAL_UPLOAD_URL")

    # check args
    # check that either args.questions_file > 1 and args.prompt_file = 1 or vice versa
    if (
        len(args.questions_file) > 1
        and len(args.prompt_file) == 1
        and len(args.output_file) > 1
    ):
        args.prompt_file = args.prompt_file * len(args.questions_file)
    elif (
        len(args.questions_file) == 1
        and len(args.prompt_file) > 1
        and len(args.output_file) > 1
    ):
        args.questions_file = args.questions_file * len(args.prompt_file)
    if not (len(args.questions_file) == len(args.prompt_file) == len(args.output_file)):
        raise ValueError(
            "If args.output_file > 1, then at least 1 of args.prompt_file or args.questions_file must be > 1 and match lengths."
            f"Obtained lengths: args.questions_file={len(args.questions_file)}, args.prompt_file={len(args.prompt_file)}, args.output_file={len(args.output_file)}"
        )

    if args.model_type == "oa":
        from runners.openai_runner import run_openai_eval

        if args.model is None:
            args.model = "gpt-3.5-turbo-0613"
        run_openai_eval(args)
    elif args.model_type == "anthropic":
        from runners.anthropic_runner import run_anthropic_eval

        if args.model is None:
            args.model = "claude-2"
        run_anthropic_eval(args)
    elif args.model_type == "vllm":
        import platform

        if platform.system() == "Darwin":
            raise ValueError(
                "vLLM is not supported on macOS. Please run on another OS supporting CUDA."
            )
        from runners.vllm_runner import run_vllm_eval

        run_vllm_eval(args)
    elif args.model_type == "hf":
        from runners.hf_runner import run_hf_eval

        run_hf_eval(args)
    elif args.model_type == "api":
        assert args.api_url is not None, "api_url must be provided for api model"
        assert args.api_type is not None, "api_type must be provided for api model"
        assert args.api_type in ["vllm", "tgi"], "api_type must be one of 'vllm', 'tgi'"

        from runners.api_runner import run_api_eval

        run_api_eval(args)
    elif args.model_type == "llama_cpp":
        from runners.llama_cpp_runner import run_llama_cpp_eval

        run_llama_cpp_eval(args)
    elif args.model_type == "mlx":
        from runners.mlx_runner import run_mlx_eval

        run_mlx_eval(args)
    elif args.model_type == "gemini":
        from runners.gemini_runner import run_gemini_eval

        run_gemini_eval(args)
    elif args.model_type == "mistral":
        from runners.mistral_runner import run_mistral_eval

        run_mistral_eval(args)
    elif args.model_type == "bedrock":
        from runners.bedrock_runner import run_bedrock_eval

        run_bedrock_eval(args)
    elif args.model_type == "together":
        from runners.together_runner import run_together_eval

        run_together_eval(args)
    elif args.model_type == "deepseek":
        from runners.deepseek_runner import run_deepseek_eval

        run_deepseek_eval(args)
    else:
        raise ValueError(
            f"Invalid model type: {args.model_type}. Model type must be one of: 'oa', 'hf', 'anthropic', 'vllm', 'api', 'llama_cpp', 'mlx', 'gemini', 'mistral'"
        )
