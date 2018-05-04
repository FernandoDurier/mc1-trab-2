import click
import csv

from preprocess import clean, p_method, n_method, p_method_extra

@click.group
def cli():
    with open('')

@cli.command
@click.argument('input_filename', type=click.Path(exists=True))
def preprocess(input_filename):
    dataset = []
    with open(input_filename) as f:
        reader = csv.reader(f)
        dataset = [row for row in reader]

    dataset = clean(dataset)


if __name__ == "__main__":
    cli()
