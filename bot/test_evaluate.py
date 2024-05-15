import json
import os
from evaluate import perform_eval, run_down_battery
from environment import BatteryEnv
import argparse
import numpy as np

def test_evaluate_single():
    args = argparse.Namespace()
    args.class_name = 'SimplePolicy'
    args.seed = 42
    args.data = 'bot/data/april15-may7_2023.csv'
    args.output_file = 'bot/results/tmp.json'
    args.param = []
    args.plot = False
    args.present_index = 0

    perform_eval(args)

    assert os.path.exists('bot/results/tmp.json') == True

    with open('bot/results/tmp.json', 'r') as file:
        data = json.load(file)

    main_trial = data['main_trial']
    assert len(main_trial['profits']) == 6335
    assert data['score'] == 2.4896999999999996

    os.remove('bot/results/tmp.json')


def test_evaluate_with_context():
    args = argparse.Namespace()
    args.class_name = 'SimplePolicy'
    args.seed = 42
    args.data = 'bot/data/april_start.csv'
    args.output_file = 'bot/results/tmp.json'
    args.param = []
    args.plot = False
    args.present_index = 0

    perform_eval(args)

    assert os.path.exists('bot/results/tmp.json') == True

    with open('bot/results/tmp.json', 'r') as file:
        data = json.load(file)

    main_trial = data['main_trial']
    assert len(main_trial['socs']) == 19
    last_profit = main_trial['profits'][-1]
    n_steps = len(main_trial['profits'])


    args = argparse.Namespace()
    args.class_name = 'SimplePolicy'
    args.seed = 42
    args.data = 'bot/data/april_next.csv'
    args.output_file = 'bot/results/tmp.json'
    args.param = []
    args.plot = False
    args.present_index = 19
    args.initial_soc = main_trial['socs'][-1]
    args.initial_profit = main_trial['profits'][-1]

    perform_eval(args)

    assert os.path.exists('bot/results/tmp.json') == True
    with open('bot/results/tmp.json', 'r') as file:
        data = json.load(file)

    main_trial = data['main_trial']

    assert len(main_trial['profits']) == 13
    assert main_trial['socs'][0] == 13
    assert main_trial['socs'][-1] == 13
    for p in main_trial['profits']:
        assert p == last_profit 

    assert n_steps + len(main_trial['profits']) == 32


    os.remove('bot/results/tmp.json')

def test_evaluate_start_step():
    args = argparse.Namespace()
    args.class_name = 'SimplePolicy'
    args.seed = 42
    args.data = 'bot/data/april15-may7_2023.csv'
    args.output_file = 'bot/results/tmp.json'
    args.param = []
    args.plot = False
    args.present_index = 100

    perform_eval(args)

    assert os.path.exists('bot/results/tmp.json') == True

    with open('bot/results/tmp.json', 'r') as file:
        data = json.load(file)

    main_trial = data['main_trial']
    assert len(main_trial['profits']) == 6235
    assert len(main_trial['actions']) == 6235
    assert '2023-04-15 00:05:00' not in main_trial['timestamps']
    assert '2023-04-15 11:50:00' in main_trial['timestamps']

    assert data['score'] == main_trial['profits'][-1] + np.sum(main_trial['rundown_profit_deltas'])

    os.remove('bot/results/tmp.json')

def test_rundown_battery():
    battery_environment = BatteryEnv(data='bot/train.csv', initial_charge_kWh=305, max_charge_rate_kW=20)
    market_prices = [10, 10, 10, 10, 10]
    run_down_battery(battery_environment, market_prices)

    assert battery_environment.battery.state_of_charge_kWh == 0