import time
import argparse
import os
import random
import pandas as pd
from datetime import datetime
import numpy as np
import json

from policies import policy_classes
from environment import BatteryEnv, PRICE_KEY, TIMESTAMP_KEY
from plotting import plot_results

def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a float or 'None'")


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)['policy']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_down_battery(battery_environment: BatteryEnv, market_prices):
    last_day_prices = market_prices[-288:]
    assumed_rundown_price = np.mean(last_day_prices)
    rundown_profits = []
    
    while battery_environment.battery.state_of_charge_kWh > 0:
        kWh_removed = battery_environment.battery.discharge_at(battery_environment.battery.max_charge_rate_kW)
        rundown_profits.append(battery_environment.kWh_to_profit(kWh_removed, assumed_rundown_price))

    return rundown_profits

def run_trial(battery_environment: BatteryEnv, policy):
    profits, socs, market_prices, battery_actions, solar_actions, pv_inputs, timestamps = [], [], [], [], [], [], []

    external_state, internal_state = battery_environment.initial_state()
    while True:
        pv_power = float(external_state["pv_power"])
        solar_kW_to_battery, charge_kW = policy.act(external_state, internal_state)

        market_prices.append(external_state[PRICE_KEY])
        timestamps.append(external_state[TIMESTAMP_KEY])
        battery_actions.append(charge_kW)
        solar_actions.append(solar_kW_to_battery)
        pv_inputs.append(pv_power)

        external_state, internal_state = battery_environment.step(charge_kW, solar_kW_to_battery, pv_power)

        profits.append(internal_state['total_profit'])
        socs.append(internal_state['battery_soc'])

        if external_state is None:
            break


    rundown_profits = run_down_battery(battery_environment, market_prices)

    return {
        'profits': profits,
        'socs': socs,
        'market_prices': market_prices,
        'actions': battery_actions,
        'solar_actions': solar_actions,
        'pv_inputs': pv_inputs,
        'final_soc': socs[-1],
        'rundown_profit_deltas': rundown_profits,
        'timestamps': timestamps
    }

def parse_parameters(params_list):
    params = {}
    for item in params_list:
        key, value = item.split('=')
        params[key] = eval(value)
    return params

def perform_eval(args):
    start = time.time()

    if args.class_name:
        policy_config = {'class_name': args.class_name, 'parameters': parse_parameters(args.param)}
    else:
        policy_config = load_config('bot/config.json')

    policy_class = policy_classes[policy_config['class_name']]
    
    external_states = pd.read_csv(args.data)
    if args.output_file:
        output_file = args.output_file
    else:
        results_dir = 'bot/results'
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{policy_config["class_name"]}.json')

    initial_profit = args.initial_profit if 'initial_profit' in args and args.initial_profit is not None else 0
    initial_soc = args.initial_soc if 'initial_soc' in args and args.initial_profit is not None else 7.5

    set_seed(args.seed)
    start_step = args.present_index

    historical_data = external_states.iloc[:start_step]
    future_data = external_states.iloc[start_step:]

    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )

    policy = policy_class(**policy_config.get('parameters', {}))
    policy.load_historical(historical_data)
    trial_data = run_trial(battery_environment, policy)

    total_profits = trial_data['profits']
    rundown_profit_deltas = trial_data['rundown_profit_deltas']

    mean_profit = float(np.mean(total_profits))
    std_profit = float(np.std(total_profits))

    mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)

    outcome = {
        'class_name': policy_config['class_name'],
        'parameters': policy_config.get('parameters', {}),
        'mean_profit': mean_profit,
        'std_profit': std_profit,
        'score': mean_combined_profit,
        'main_trial': trial_data,
        'seconds_elapsed': time.time() - start 
    }

    print(f'Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}')
    print(f'Average profit inc rundown ($): {mean_combined_profit:.2f}')

    with open(output_file, 'w') as file:
        json.dump(outcome, file, indent=2)

    if args.plot:
        plot_results(trial_data['profits'], trial_data['market_prices'], trial_data['socs'], trial_data['actions'])

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single energy market strategy.')
    parser.add_argument('--plot', action='store_true', help='Plot the results of the main trial.', default=False)
    parser.add_argument('--present_index', type=int, default=0, help='Index to split the historical data from the data which will be used for the evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomness')
    parser.add_argument('--data', type=str, default='bot/data/april15-may7_2023.csv', help='Path to the market data csv file')
    parser.add_argument('--class_name', type=str, help='Policy class name. If not provided, the config.json policy will be used.')
    parser.add_argument('--output_file', type=str, help='File to save all the submission outputs to.', default=None)
    parser.add_argument('--param', action='append', help='Policy parameters as key=value pairs', default=[])
    parser.add_argument('--initial_soc', type=float_or_none, help='Initial state of charge of the battery in kWh', default=None)
    parser.add_argument('--initial_profit', type=float_or_none, help='Initial profit of the battery in $', default=None)

    args = parser.parse_args()

    perform_eval(args)

if __name__ == '__main__':
    main()