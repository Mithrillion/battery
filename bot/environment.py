"""
This file contains the core functionality for the battery simulation environment and its interaction with the NEM.
It provides a realistic representation of the NEM's operations and the battery's role within it.
Contestants will use this environment to develop and test their bidding strategies.

Units used in this file:
- Power: kilowatts (kW)
- Energy: kilowatt-hours (kWh)
- Time: minutes (min)
- Price: dollars per kilowatt-hour ($/kWh)
"""

import pandas as pd
import numpy as np
from collections import deque
from typing import Tuple

INTERVAL_DURATION = 5  # Duration of each dispatch interval in minutes
PRICE_KEY = 'price'
TIMESTAMP_KEY = 'timestamp'

def kWh_to_kW(kWh: float) -> float:
    """
    Convert energy in kilowatt-hours (kWh) to power in kilowatts (kW).

    :param kWh: Energy in kilowatt-hours (kWh).
    :return: Power in kilowatts (kW).
    """
    return kWh / (INTERVAL_DURATION / 60)

def kW_to_kWh(kWh: float) -> float:
    """
    Convert energy in kilowatt-hours (kWh) to power in kilowatts (kW).

    :param kWh: Energy in kilowatt-hours (kWh).
    :return: Power in kilowatts (kW).
    """
    return kWh * (INTERVAL_DURATION / 60)

class Battery:
    def __init__(self, capacity_kWh: float, max_charge_rate_kW: float, initial_charge_kWh: float):
        self.capacity_kWh = capacity_kWh
        self.initial_charge_kWh = initial_charge_kWh
        self.max_charge_rate_kW = max_charge_rate_kW
        self._state_of_charge_kWh = min(self.initial_charge_kWh, self.capacity_kWh)

    def reset(self):
        self._state_of_charge_kWh = min(self.initial_charge_kWh, self.capacity_kWh)
    
    def charge_at(self, kW: float) -> float:
        kW = min(kW, self.max_charge_rate_kW)
        kWh_to_add = kW_to_kWh(kW) 
        kWh_to_add = min(kWh_to_add, self.capacity_kWh - self._state_of_charge_kWh)
        self._state_of_charge_kWh += kWh_to_add
        return kWh_to_add

    def discharge_at(self, kW: float) -> float:
        kW = min(kW, self.max_charge_rate_kW)
        kW_to_remove = kW_to_kWh(kW)
        kW_to_remove = min(kW_to_remove, self._state_of_charge_kWh)
        self._state_of_charge_kWh = max(self._state_of_charge_kWh - kW_to_remove, 0)
        return kW_to_remove

    @property
    def state_of_charge_kWh(self) -> float:
        return self._state_of_charge_kWh


class BatteryEnv:
    """
    Environment for simulating battery operation in the National Electricity Market (NEM) context.
    """
    def __init__(self, data, capacity_kWh: float = 13, max_charge_rate_kW: float = 5, initial_charge_kWh: float = 7.5, initial_profit: float = 0.0):
        self.battery = Battery(capacity_kWh, max_charge_rate_kW, initial_charge_kWh)
        self.market_data = data
        self.total_profit = initial_profit
        self.current_step = 0
        self.episode_length = len(self.market_data)

    def initial_state(self):
        assert self.current_step == 0

        return self.market_data.iloc[self.current_step], self.get_info(0)

    def step(self, charge_kW: float, solar_kW_to_battery:int, total_solar_kW:int) -> Tuple[pd.Series, dict]:
        if self.current_step >= len(self.market_data):
            return None, None
        market_price_mWh = self.market_data.iloc[self.current_step][PRICE_KEY]

        kW_currently_charging, solar_profit_delta = self.process_solar(solar_kW_to_battery, total_solar_kW, market_price_mWh)

        max_charge_kW = self.battery.max_charge_rate_kW - kW_currently_charging
        battery_profit_delta = self.charge_discharge(min(charge_kW, max_charge_kW), market_price_mWh)
        
        external_state = self.get_info(battery_profit_delta + solar_profit_delta)

        self.current_step += 1
        if self.current_step >= len(self.market_data):
            return None, external_state

        return self.market_data.iloc[self.current_step], external_state

    def process_solar(self, solar_kW_to_battery: int, total_solar_kW: int, market_price_mWh:int) -> float:
        solar_kW_to_battery = max(0, min(total_solar_kW, solar_kW_to_battery))

        kWh_charged = self.battery.charge_at(solar_kW_to_battery)
        kW_charging = kWh_to_kW(kWh_charged)
        energy_to_grid_kWh = kW_to_kWh(total_solar_kW) - kWh_charged
        profit = self.kWh_to_profit(energy_to_grid_kWh, market_price_mWh)

        return kW_charging, profit

    def kWh_to_profit(self, energy_removed: float, spot_price_mWh: float) -> float:
        return round(energy_removed * spot_price_mWh / 1000, 4)

    def charge_discharge(self, charge_kW: float, spot_price_mWh: float) -> float:
        if charge_kW > 0:
            kWh_to_battery = self.battery.charge_at(charge_kW)
            return -self.kWh_to_profit(kWh_to_battery, spot_price_mWh) 
        elif charge_kW < 0:
            kWh_to_grid = self.battery.discharge_at(-charge_kW)
            return self.kWh_to_profit(kWh_to_grid, spot_price_mWh)
        return 0

    def get_info(self, profit_delta: float = 0) -> dict:
        """
        Return a dictionary containing relevant information for the agent.

        :param profit_delta: The change in profit from the last action (default: 0).
        :return: A dictionary containing information about the current state of the environment.
        """
        self.total_profit += profit_delta
        remaining_steps = len(self.market_data) - self.current_step - 1
        return {
            'total_profit': self.total_profit,
            'profit_delta': profit_delta,
            'battery_soc': self.battery.state_of_charge_kWh,
            'max_charge_rate': self.battery.max_charge_rate_kW,
            'remaining_steps': remaining_steps
        }