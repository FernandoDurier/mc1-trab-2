import click
import csv

from preprocess import clean, p_method, n_method
from evaluate import metrics, correct, hypothesis_tests
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
    #sampled_dataset = sample(dataset, 20)
    #print("Result of N_Method:",n_method(sampled_dataset, 5))
    #print("Result of P_Method:",p_method(sampled_dataset, 5))
    #print("Testing the n method against the p method:",kolgomorov2samples(n_method(sampled_dataset, 5),p_method(sampled_dataset, 5)))
    #print("Test Z between methods n and p:",testz(n_method(sampled_dataset, 5),p_method(sampled_dataset, 5)))

    gs = []
    with open(gs_filename) as f:
        reader = csv.reader(f)
        gs = [row for row in reader]

    y_true = correct(clean(gs), 3)
    
    output = []
    for crowd_size in range(20, 81):
        p, r, f = metrics(dataset, y_true, crowd_size)
        output.append([crowd_size, 'precision'] + hypothesis_tests(p).tolist())
        output.append([crowd_size, 'recall'] + hypothesis_tests(r).tolist())
        output.append([crowd_size, 'f_measure'] + hypothesis_tests(f).tolist())

    with open('output.csv', 'w') as f:
        writer = csv.writer(f, delimiter=';')
        for line in output:
            writer.writerow(line)

if __name__ == "__main__":
    cli()
