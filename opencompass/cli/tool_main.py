# flake8: noqa
import argparse
import sys

from ..tools import (case_analyzer, collect_code_preds, compare_configs,
                     convert_alignmentbench, list_configs, prediction_merger,
                     prompt_viewer, test_api_model, update_dataset_suffix,
                     viz_multi_model)

# 定义所有可用的子命令模块
TOOLS = {
    'list_configs': list_configs,
    # 'prompt_viewer': prompt_viewer,
    # 'case_analyzer': case_analyzer,
    # 'collect_code_preds': collect_code_preds,
    # 'compare_configs': compare_configs,
    # 'convert_alignmentbench': convert_alignmentbench,
    # 'prediction_merger': prediction_merger,
    # 'test_api_model': test_api_model,
    # 'update_dataset_suffix': update_dataset_suffix,
    # 'viz_multi_model': viz_multi_model,
}


def main():
    parser = argparse.ArgumentParser(
        description='A toolset for various operations.')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True  # Make subparsers required

    # Create subparsers for each module
    for name, module in TOOLS.items():
        subparser = subparsers.add_parser(name, help=module.__doc__)
        # Assume each module has a parse_args function to handle arguments
        module.parse_args(subparser)
        subparser.set_defaults(func=module.main)

    args = parser.parse_args()

    # If the user requests specific subcommand help information
    if '--help' in sys.argv[1:]:
        if len(sys.argv) > 2 and sys.argv[1] in TOOLS:
            command = sys.argv[1]
            module = TOOLS[command]
            module.parse_args(['--help'])
        else:
            parser.print_help()
    else:
        # Otherwise execute the subcommand
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()


if __name__ == '__main__':
    main()
