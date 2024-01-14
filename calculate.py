import argparse
from pathlib import Path
import sys
import json
import pandas as pd


def read_own_usage(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize(df):
    # normalizes data points to year basis
    constant_day_mask = df.constant_unit == 'day'
    variable_day_mask = df.variable_unit == 'day'

    constant_month_mask = df.constant_unit == 'month'
    variable_month_mask = df.variable_unit == 'month'

    df = df[constant_day_mask | variable_day_mask] * 365.0f
    df = df[constant_month_mask | variable_month_mask] * 365.0f
    return df
    

def best_offer(df, own):
    # Calculates the best contract for power and gas. Considers both combined and separate contracts

    # combined

    # separate


def main():
    # Simple program to calculate normalized cost for power & gas grid connections.
    parser = argparse.ArgumentParser()
    parser.add_argument('--offers', default=str(Path.cwd() / 'offers.csv'), help='Data file containing offers.')
    parser.add_argument('--own', default=str(Path.cwd() / 'own.json'), help='Data file containing own usage.')
    parser.add_argument('--test', action='store_true', help='Sets test mode')
    args = parser.parse_args()

    own = read_own_usage(args)
    df = pd.read_csv(args.from)
    df = normalize(df)
    print(best_offer(df, own))
    

if __name__ == '__main__':
    main()
