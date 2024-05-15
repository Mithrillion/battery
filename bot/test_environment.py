import numpy as np
import pandas as pd
from bot.environment import BatteryEnv, Battery, INTERVAL_DURATION, kW_to_kWh, kWh_to_kW

PATH_TO_DATA = 'bot/data/april15-may7_2023.csv'

def test_battery_environment():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    count = 0
    while True:
        quantity = 1

        state, info = battery_env.step(quantity, 0, 0)

        count += 1
        if state is None:
            break
    
    assert count == len(data)
    
    duration_in_hours = INTERVAL_DURATION / 60
    energy_MWh = quantity * duration_in_hours / 1000
    expected_total_profit = -(
        data["price"] * energy_MWh
    ).round(7).sum()
    assert battery_env.total_profit < 0
    assert np.isclose(battery_env.total_profit, expected_total_profit, atol=1e-1)


def test_battery_env_one_step():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(3, 0, 0)

    assert info['battery_soc'] == 7.5 + 3 * INTERVAL_DURATION / 60 
    assert info['profit_delta'] == 0.0117

def test_conversion_kWh_to_kW():
    energy = 10
    power = kWh_to_kW(energy)
    energy = kW_to_kWh(power)
    assert energy == 10

def test_battery_env_double_charge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    state, info = battery_env.step(3, 3, 3)

    assert info['battery_soc'] == 7.5 + (3 + 2) * INTERVAL_DURATION / 60, info['battery_soc']
    assert info['profit_delta'] == 0.0078

def test_battery_env_double_charge2():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=4,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    state, info = battery_env.step(3, 3, 3)

    assert info['battery_soc'] == 7.5 + (3 + 1) * INTERVAL_DURATION / 60, info['battery_soc']
    assert info['profit_delta'] == 0.0039


def test_battery_env_double_charge3():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=2,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    state, info = battery_env.step(3, 3, 3)

    assert info['battery_soc'] == 7.5 + 2 * INTERVAL_DURATION / 60, info['battery_soc']
    assert info['profit_delta'] == -0.0039

def test_battery_env_solar_and_discharge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 3, 3)

    assert info['battery_soc'] == 7.5
    assert info['profit_delta'] == -0.0117


def test_battery_env_double_discharge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 0, 3)

    assert info['battery_soc'] == 7.25
    assert info['profit_delta'] == -0.0234


def test_battery_env_negatives():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, -4, 3)

    assert info['battery_soc'] == 7.25
    assert info['profit_delta'] == -0.0234

def test_battery_env_double_discharge_empty():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 0, 3)

    assert info['battery_soc'] == 0
    assert info['profit_delta'] == -0.0117


def test_battery_env_double_discharge_empty():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 3, 3)

    assert info['battery_soc'] == 0
    assert info['profit_delta'] == -0.0117

def test_battery_env_too_high():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 7, 3)

    assert info['battery_soc'] == 0
    assert info['profit_delta'] == -0.0117


def test_battery_env_double_discharge_empty_less_solar():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 2, 2)

    assert info['battery_soc'] == 0
    assert info['profit_delta'] == -0.0078


def test_battery_env_double_discharge_no_pv():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=4)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 2, 0)

    assert info['battery_soc'] == 3.75
    assert info['profit_delta'] == -0.0117

def test_battery_env_too_much_solar():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-3, 3, 2)

    assert info['battery_soc'] == 0
    assert info['profit_delta'] == -0.0078

def test_battery_env_double_discharge_empty_less_discharge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=0)
    state, info = battery_env.initial_state()
    state, info = battery_env.step(-2, 3, 3)

    assert (info['battery_soc'] - 0.0833333) < 0.01
    assert info['profit_delta'] == -0.0078

def test_battery_env_battery_and_solar():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    state, info = battery_env.step(3, 2, 3)

    assert info['battery_soc'] == 7.5 + (2 + 3) * INTERVAL_DURATION / 60, info['battery_soc']
    # test approximate profit using absolute difference
    assert abs(info['profit_delta'] - 0.0078) < 0.0001


def test_battery_env_solar_to_battery():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()


    state, info = battery_env.step(3, 0.5, 3)

    assert info['battery_soc'] == 7.5 + (0.5 + 3) * INTERVAL_DURATION / 60, info['battery_soc']
    assert abs(info['profit_delta'] -  0.0019) < 0.0001


def test_battery_env_all_to_battery():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=13, max_charge_rate_kW=5,  initial_charge_kWh=7.5)
    state, info = battery_env.initial_state()

    state, info = battery_env.step(3, 0, 3)

    assert info['battery_soc'] == 7.5 + 3 * INTERVAL_DURATION / 60, info['battery_soc']
    assert info['profit_delta'] == 0

def test_battery_env_full_battery():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=13, max_charge_rate_kW=5,  initial_charge_kWh=13)
    state, info = battery_env.initial_state()
    print('price', state['price'])

    state, info = battery_env.step(3, 0, 3)

    assert info['battery_soc'] == 13
    assert info['profit_delta'] == -0.0117

def test_battery_env_full_battery_double_charge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=13, max_charge_rate_kW=5,  initial_charge_kWh=13)
    state, info = battery_env.initial_state()
    print('price', state['price'])

    state, info = battery_env.step(3, 3, 3)

    assert info['battery_soc'] == 13
    assert info['profit_delta'] == -0.0117

def test_price_for_single_discharge():
    data = pd.read_csv(PATH_TO_DATA)
    battery_env = BatteryEnv(data=data, capacity_kWh=10000, max_charge_rate_kW=5,  initial_charge_kWh=7.5)

    energy = 5
    spot_price_MWh = 100
    profit = battery_env.kWh_to_profit(energy, spot_price_MWh)
    expected_profit = energy * spot_price_MWh / 1000

    assert np.isclose(profit, expected_profit)

def test_battery_charges_once():
    battery = Battery(capacity_kWh=13, max_charge_rate_kW=5, initial_charge_kWh=0)
    charge = battery.charge_at(5)
    assert abs(charge - 0.41666666666666663) < 0.01
    assert abs(battery.state_of_charge_kWh - 0.41666666666666663) < 0.01

    charge = battery.charge_at(5)
    assert abs(charge - 0.41666666666666663) < 0.01
    assert abs(battery.state_of_charge_kWh - (2* 0.41666666666666663)) < 0.01