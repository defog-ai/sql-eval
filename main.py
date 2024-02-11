import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions_file", type=str)
    parser.add_argument("-n", "--num_questions", type=int, default=None)
    parser.add_argument("-db", "--db_type", type=str, required=True)
    parser.add_argument("-g", "--model_type", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-a", "--adapter", type=str)
    parser.add_argument("--api_url", type=str)
    parser.add_argument("-b", "--num_beams", type=int, default=4)
    # take in a list of prompt files
    parser.add_argument("-f", "--prompt_file", nargs="+", type=str, required=True)
    parser.add_argument("-d", "--use_private_data", action="store_true")
    parser.add_argument("-k", "--k_shot", action="store_true")
    parser.add_argument("-o", "--output_file", nargs="+", type=str, required=True)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--upload_url", type=str)
    parser.add_argument(
        "-qz", "--quantized", default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    # if questions_file is None, set it to the default questions file for the given db_type
    if args.questions_file is None:
        args.questions_file = f"data/questions_gen_{args.db_type}.csv"

    # check that questions_file matches db_type
    if args.db_type not in args.questions_file:
        print(
            f"WARNING: Check that questions_file {args.questions_file} is compatible with db_type {args.db_type}"
        )

    # check that the list of prompt files has the same length as the list of output files
    if len(args.prompt_file) != len(args.output_file):
        raise ValueError(
            f"Number of prompt files ({len(args.prompt_file)}) must be the same as the number of output files ({len(args.output_file)})"
        )

    if args.model_type == "oa":
        from eval.openai_runner import run_openai_eval

        if args.model is None:
            args.model = "gpt-3.5-turbo-0613"
        run_openai_eval(args)
    elif args.model_type == "anthropic":
        from eval.anthropic_runner import run_anthropic_eval

        if args.model is None:
            args.model = "claude-2"
        run_anthropic_eval(args)
    elif args.model_type == "vllm":
        import platform

        if platform.system() == "Darwin":
            raise ValueError(
                "vLLM is not supported on macOS. Please run on another OS supporting CUDA."
            )
        from eval.vllm_runner import run_vllm_eval

        run_vllm_eval(args)
    elif args.model_type == "hf":
        from eval.hf_runner import run_hf_eval

        run_hf_eval(args)
    elif args.model_type == "api":
        from eval.api_runner import run_api_eval

        run_api_eval(args)
    elif args.model_type == "llama_cpp":
        from eval.llama_cpp_runner import run_llama_cpp_eval

        run_llama_cpp_eval(args)
    else:
        raise ValueError(
            f"Invalid model type: {args.model_type}. Model type must be one of: 'oa', 'hf', 'anthropic', 'vllm', 'api'"
        )
