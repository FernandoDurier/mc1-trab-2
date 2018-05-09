import click
import csv

from preprocess import clean, p_method, n_method
from evaluate import sample, metrics, correct, hypothesis_tests
from tests import kolgomorov2samples, testz

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_filename', type=click.Path(exists=True))
@click.argument('gs_filename', type=click.Path(exists=True))
def process(input_filename, gs_filename):
    dataset = []
    with open(input_filename) as f:
        reader = csv.reader(f)
        dataset = [row for row in reader]

    dataset = clean(dataset)
    sampled_dataset = sample(dataset, 20)
    #print("Result of N_Method:",n_method(sampled_dataset, 5))
    #print("Result of P_Method:",p_method(sampled_dataset, 5))
    #print("Testing the n method against the p method:",kolgomorov2samples(n_method(sampled_dataset, 5),p_method(sampled_dataset, 5)))
    #print("Test Z between methods n and p:",testz(n_method(sampled_dataset, 5),p_method(sampled_dataset, 5)))

    gs = []
    with open(gs_filename) as f:
        reader = csv.reader(f)
        gs = [row for row in reader]

    y_true = correct(clean(gs), 3)
    

    for sample_size in range(20, 80):
        p, r, f = metrics(dataset, y_true, 20)
        print(f"Sample size = {sample_size}")
        print("Results for precision:", hypothesis_tests(p))
        print("Results for recall:", hypothesis_tests(r))
        print("Results for f1_score:", hypothesis_tests(f))


if __name__ == "__main__":
    cli()
