import click
import csv

from preprocess import clean, p_method, n_method
from tests import kolgomorov2samples, testz

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_filename', type=click.Path(exists=True))
def preprocess(input_filename):
    dataset = []
    with open(input_filename) as f:
        reader = csv.reader(f)
        dataset = [row for row in reader]

    dataset = clean(dataset)
    print("Result of N_Method:",n_method(dataset, 5))
    print("Result of P_Method:",p_method(dataset, 5))
    print("Testing the n method against the p method:",kolgomorov2samples(n_method(dataset, 5),p_method(dataset, 5)))
    print("Test Z between methods n and p:",testz(n_method(dataset, 5),p_method(dataset, 5)))
if __name__ == "__main__":
    cli()
