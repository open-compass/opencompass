import argparse

import tabulate

from opencompass.utils import match_files


def parse_args(parser):
    parser.add_argument(
        'pattern',
        nargs='*',
        default=['*'],
        type=str,
        help='Patterns, wildcard matching supported.',
    )
    return parser


def main(args):
    models = match_files('opencompass/configs/models/',
                         args.pattern,
                         fuzzy=True)
    if models:
        table = [['Model', 'Config Path'], *models]
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='psql'))
    datasets = match_files('opencompass/configs/datasets/',
                           args.pattern,
                           fuzzy=True)
    if datasets:
        table = [['Dataset', 'Config Path'], *datasets]
        print(tabulate.tabulate(table, headers='firstrow', tablefmt='psql'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Utils to list available models and datasets.')
    parser = parse_args(parser)
    args = parser.parse_args()
    main(args)
