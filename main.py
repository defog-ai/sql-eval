from eval import runner
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-q", "--questions_file", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-g", "--qg_class", type=str, required=True)
    parser.add_argument("-f", "--prompt_file", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("-n", "--num_questions", type=int, default=-1)
    parser.add_argument("-p", "--parallel_threads", type=int, default=5)
    parser.add_argument("-t", "--timeout_gen", type=float, default=30.0)
    parser.add_argument("-u", "--timeout_exec", type=float, default=10.0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    runner.run(args)
