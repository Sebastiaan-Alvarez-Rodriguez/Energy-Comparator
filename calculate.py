import argparse
from pathlib import Path
import functools
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def set_if_not_null(d, key, value):
    if value:
        d[key] = value

def get_profile(path=None, profile_gas_mean=None, profile_gas_dev=None, profile_power_low_mean=None, profile_power_low_dev=None, profile_power_high_mean=None, profile_power_high_dev=None, profile_power_gen_low_mean=None, profile_power_gen_low_dev=None, profile_power_gen_high_mean=None, profile_power_gen_high_dev=None):

    if path:
        with open(path, 'r') as f:
            profile = json.load(f)
    else:
        profile = {}

    set_if_not_null(profile, 'gas_mean', profile_gas_mean)
    set_if_not_null(profile, 'gas_dev', profile_gas_dev)
    set_if_not_null(profile, 'power_low_mean', profile_power_low_mean)
    set_if_not_null(profile, 'power_low_dev', profile_power_low_dev)
    set_if_not_null(profile, 'power_high_mean', profile_power_high_mean)
    set_if_not_null(profile, 'power_high_dev', profile_power_high_dev)
    set_if_not_null(profile, 'power_gen_low_mean', profile_power_gen_low_mean)
    set_if_not_null(profile, 'power_gen_low_dev', profile_power_gen_low_dev)
    set_if_not_null(profile, 'power_gen_high_mean', profile_power_gen_high_mean)
    set_if_not_null(profile, 'power_gen_high_dev', profile_power_gen_high_dev)
    for name in ['gas', 'power_low', 'power_high', 'power_gen_low', 'power_gen_high']:
        if f'{name}_mean' not in profile:
            print(f'Key "{name}" mean not set, assuming {name}_mean = 0.0')
            profile[f'{name}_mean'] = 0.0
        if f'{name}_dev' not in profile:
            print(f'Key "{name}" standard deviation not set.')
            profile[f'{name}_dev'] = 0.0
    return profile


def normalize(df):
    # normalizes data points to year basis
    power_month_mask = df.powerconstant_unit == 'month'
    df['powerconstant'] = df.powerconstant * 12 * power_month_mask + df.powerconstant * (~power_month_mask)
    power_day_mask = df.powerconstant_unit == 'day'
    df['powerconstant'] = df.powerconstant * 12 * power_day_mask + df.powerconstant * (~power_day_mask)
    df['powerconstant_unit'] = 'year'
    gas_month_mask = df.gasconstant_unit == 'month'
    df['gasconstant'] = df.gasconstant * 12 * gas_month_mask + df.gasconstant * (~gas_month_mask)
    gas_day_mask = df.gasconstant_unit == 'day'
    df['gasconstant'] = df.gasconstant * 12 * gas_day_mask + df.gasconstant * (~gas_day_mask)
    df['gasconstant_unit'] = 'year'
    return df


def best_offer(df, gas, power_low, power_high, power_gen_low, power_gen_high, saldering, outage):
    # Calculates the best contract for power and gas. Considers both combined and separate contracts

    df['calc_gas'] = df.gasconstant + gas * df.gasvariable
    expected_low = power_low-(power_gen_low*saldering*outage)
    expected_high = power_high-(power_gen_high*saldering*outage)
    df['calc_power'] = df.powerconstant + \
        (expected_low * df.powervariable_low if expected_low > 0 else expected_low * df.powergen_low) + \
        (expected_high * df.powervariable_high if expected_high > 0 else expected_high * df.powergen_high) 

    df['total'] = df.calc_gas.fillna(0) + df.calc_power.fillna(0) + df.bonus.fillna(0)

    only_gas_mask = df.powerconstant.isna() & df.powervariable_low.isna() & df.powervariable_high.isna()
    only_power_mask = df.gasconstant.isna() & df.gasvariable.isna()
    # combined
    combined_df = df[(~only_gas_mask) & (~only_power_mask)]
    min_combined_idx = combined_df.loc[combined_df.total.idxmin()]
    # print(f'combined: {min_combined_idx}')

    # separate
    only_gas_df = df[only_gas_mask]
    min_gas_idx = only_gas_df.loc[only_gas_df.total.idxmin()]

    only_power_df = df[only_power_mask]
    min_power_idx = only_power_df.loc[only_power_df.total.idxmin()]
    # print(f'individual gas: {min_gas_idx}')
    # print(f'individual power idx: {min_power_idx}')
    if min_combined_idx.total <= min_power_idx.total + min_gas_idx.total:
        return pd.Series([min_combined_idx['name'], min_combined_idx.type, min_combined_idx.gasconstant, min_combined_idx.gasconstant_unit, min_combined_idx.gasvariable, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, min_combined_idx.bonus / 2, min_combined_idx.calc_gas, np.nan, min_combined_idx.total / 2], index=df.columns), \
        pd.Series([min_combined_idx['name'], min_combined_idx.type, np.nan, np.nan, np.nan, min_combined_idx.powerconstant, min_combined_idx.powerconstant_unit, min_combined_idx.powervariable_low, min_combined_idx.powervariable_high, min_combined_idx.powergen_low, min_combined_idx.powergen_high, min_combined_idx.bonus / 2, np.nan, min_combined_idx.calc_power, min_combined_idx.total / 2], index=df.columns)
    return min_gas_idx, min_power_idx


def find_minimal_function(df, mean_vector, cov_matrix, saldering, outage, samplesize=1000):
    # Find the linear function with the highest probability for each set of mean and covariance
    gas_dict = {}
    power_dict = {}
    linear_func = functools.partial(best_offer, df, saldering=saldering, outage=outage)
    inputs = np.random.multivariate_normal(mean_vector, np.diagflat(cov_matrix), samplesize)
    outputs = [linear_func(*sample) for sample in inputs]

    for gas, power in outputs:
        gas_dict.setdefault(gas['name'], []).append(gas.total)
        power_dict.setdefault(power['name'], []).append(power.total)
    for k, v in gas_dict.items():
        gas_dict[k] = (len(v), np.mean(v))
    for k, v in power_dict.items():
        power_dict[k] = (len(v), np.mean(v))

    best_gas = max(gas_dict, key=lambda x: gas_dict[x][0]) # gets the key with most occurrences (=most optimal values)
    best_power = max(power_dict, key=lambda x: power_dict[x][0])
    print('On average, the best contract is:')
    print(f'For gas: "{best_gas}" with a price of {gas_dict[best_gas][1]} euro per year, {gas_dict[best_gas][1]/12} euro per month (best in {gas_dict[best_gas][0]/samplesize *100}% of situations measured)')
    print(f'For power: "{best_power}" with a price of {power_dict[best_power][1]} euro per year, {power_dict[best_power][1]/12} euro per month (best in {power_dict[best_power][0]/samplesize *100}% of situations measured)')
    print()
    print('Case summary:') # TODO: normalize each dict-value list, and compare the normal equations.
    print(gas_dict)
    print(power_dict)

def main():
    # Simple program to calculate normalized cost for power & gas grid connections.
    parser = argparse.ArgumentParser()
    parser.add_argument('--offers', default=str(Path.cwd() / 'offers.csv'), help='Data file containing offers.')
    parser.add_argument('--profile', default=None, help='Data file containing usage and generation data.')
    parser.add_argument('--profile-gas-mean', type=float, default=None, help='Average yearly gas consumption. Takes precedence over file-based data.')
    parser.add_argument('--profile-gas-dev', type=float, default=None, help='Deviation for yearly gas consumption. Takes precedence over file-based data.')
    parser.add_argument('--profile-power-low-mean', type=float, default=None, help='Average yearly power consumption (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-low-dev', type=float, default=None, help='Deviation for yearly power consumption (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-high-mean', type=float, default=None, help='Average yearly power consumption (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-high-dev', type=float, default=None, help='Deviation for yearly power consumption (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-gen-low-mean', type=float, default=None, help='Average yearly power generation (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-gen-low-dev', type=float, default=None, help='Deviation for yearly power generation (low tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-gen-high-mean', type=float, default=None, help='Average yearly power generation (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--profile-power-gen-high-dev', type=float, default=None, help='Deviation for yearly power generation (high tarif). Takes precedence over file-based data.')
    parser.add_argument('--saldering', type=float, default=1.0, help='Saldering assumption, determines how much generated energy compensates for consumed energy (factor between 0 and 1).')
    parser.add_argument('--outage', type=float, default=1.0, help='Outage assumption, describes when the sun shines but the net is satisfied, leading to no power delivery on the net (factor between 0 and 1).')
    parser.add_argument('--extra-costs', type=float, default=0.0, help='Extra: Other costs, can be negative to denote a bonus.')
    parser.add_argument('--test', action='store_true', help='Sets test mode')
    args = parser.parse_args()

    profile = get_profile(args.profile, \
                          args.profile_gas_mean, args.profile_gas_dev, \
                          args.profile_power_low_mean, args.profile_power_low_dev, \
                          args.profile_power_high_mean, args.profile_power_high_dev, \
                          args.profile_power_gen_low_mean, args.profile_power_gen_low_dev, \
                          args.profile_power_gen_high_mean, args.profile_power_gen_high_dev)
    df = pd.read_csv(args.offers)
    print(df.to_string())
    df = normalize(df)

    mean_vector = [args.profile_gas_mean, args.profile_power_low_mean, args.profile_power_high_mean, args.profile_power_gen_low_mean, args.profile_power_gen_high_mean]
    std_vector = [args.profile_gas_dev, args.profile_power_low_dev, args.profile_power_high_dev, args.profile_power_gen_low_dev, args.profile_power_gen_high_dev]

    if all(v == 0.0 for v in std_vector):
        gas, power = best_offer(df, profile['gas_mean'], profile['power_low_mean'], profile['power_high_mean'], profile['power_gen_low_mean'], profile['power_gen_high_mean'], args.saldering, args.outage)
        year_price = gas.total+power.total + args.extra_costs
        print('No variability detected.')
        print(f'The best offer is for {year_price} euro per year, or {year_price/12} euro per month:')
        print('Gas:')
        print(gas)
        print('Power:')
        print(power)
    else:
        if 0.0 in std_vector:
            print('found at least one standard deviation to be 0. Defaulting to 0.1, indicating almost no variance (nonzero required for SVD algorithm)')
        std_vector = [v if v != 0.0 else 0.1 for v in std_vector]
        find_minimal_function(df, mean_vector, std_vector, args.saldering, args.outage)

if __name__ == '__main__':
    main()
