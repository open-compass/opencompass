import argparse

from ..tools import (case_analyzer, collect_code_preds, compare_configs,
                     convert_alignmentbench, list_configs, prediction_merger,
                     prompt_viewer, test_api_model, update_dataset_suffix,
                     viz_multi_model)

# 定义所有可用的子命令模块
TOOLS = {
    'list-configs': list_configs,
    'prompt-viewer': prompt_viewer,
    'case-analyzer': case_analyzer,
    'collect-code-preds': collect_code_preds,
    'compare-configs': compare_configs,
    'convert-alignmentbench': convert_alignmentbench,
    'prediction-merger': prediction_merger,
    'test-api-model': test_api_model,
    'update-dataset-suffix': update_dataset_suffix,
    'viz-multi-model': viz_multi_model,
}


def main():
    parser = argparse.ArgumentParser(
        description='A toolset for various operations.')
    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True  # Ensure a subcommand is specified

    # Create subparsers for each module
    for name, module in TOOLS.items():
        subparser = subparsers.add_parser(name, help=module.__doc__)
        # Get the ArgumentParser object for the subcommand, add its arguments
        module.parse_args(subparser)
        subparser.set_defaults(func=module.main)

    args = parser.parse_args()

    # Execute the subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
