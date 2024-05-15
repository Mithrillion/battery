from typing import Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

INTERVAL_DURATION = 5  # Duration of each dispatch interval in minutes
PRICE_KEY = "price"
TIMESTAMP_KEY = "timestamp"

# Conversion functions from the official script


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


class BatteryEnv(nn.Module):
    def __init__(
        self,
        capacity_kWh: float,
        max_charge_rate_kW: float,
        initial_charge_kWh: float,
        round_profit: bool = False,
    ):
        """
        Initializes a battery environment.

        :param capacity_kWh: The capacity of the battery in kilowatt-hours.
        :param max_charge_rate_kW: The maximum charge rate of the battery in kilowatts.
        :param initial_charge_kWh: The initial charge of the battery in kilowatt-hours.
        """
        super().__init__()
        self.capacity_kWh = capacity_kWh
        self.initial_charge_kWh = initial_charge_kWh
        assert self.initial_charge_kWh <= self.capacity_kWh
        self.max_charge_rate_kW = max_charge_rate_kW
        self.round_or_not = (
            partial(torch.round, decimals=4) if round_profit else lambda x: x
        )

    @staticmethod
    def soft_clamp_func(x, min_val, max_val, beta):
        return min_val + F.softplus(x - min_val, beta) - F.softplus(x - max_val, beta)

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """
        Get the initial state of the battery.

        :param batch_size: The batch size.
        :return: The initial state of the battery.
        """
        return torch.full((batch_size,), self.initial_charge_kWh)[:, None]

    def get_random_initial_state(self, batch_size: int) -> torch.Tensor:
        """
        Get the initial state of the battery.

        :param batch_size: The batch size.
        :return: The initial state of the battery.
        """
        return torch.rand((batch_size, 1)) * self.capacity_kWh

    # in this version of battery env, this funciton takes one action at a time and compute the cost and battery state after that action
    # this is different from the batch version of the battery env
    def forward(
        self,
        current_charge_kWh: torch.Tensor,
        grid_action: torch.Tensor,
        pv_action: torch.Tensor,
        pv_power: torch.Tensor,
        price: torch.Tensor,
        beta: float = None,
        is_peak_time_if_taxed: Optional[torch.Tensor] = None,
        output_debug_info: bool = False,
        input_raw_kWh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the given action to the battery.

        :param current_charge_kWh: The current charge of the battery in kilowatt-hours.
        :param grid_action: Grid action.
        :param pv_action: PV action.
        :param pv_power: PV power.
        :param price: Price.
        :param beta: Beta value for softplus function (optional).
        :param is_peak_if_taxed: Whether the current time is peak time if taxed (optional).
        :return: Tuple containing the new battery state and cost.
        """

        if beta is not None:
            clamp_func = partial(self.soft_clamp_func, beta=beta)
        else:
            clamp_func = torch.clamp

        # process solar first
        attempted_pv_charge_rate = (
            pv_action * pv_power if not input_raw_kWh else pv_action
        )
        realised_pv_charge_rate = clamp_func(
            attempted_pv_charge_rate, 0, self.max_charge_rate_kW
        )
        provisional_pv_charge_amount = kW_to_kWh(realised_pv_charge_rate)

        # process grid next
        attempted_grid_charge_rate = (
            grid_action * self.max_charge_rate_kW if not input_raw_kWh else grid_action
        )

        # charge the battery with pv first
        realised_pv_charge_amount = (
            clamp_func(
                current_charge_kWh + provisional_pv_charge_amount,
                0,
                self.capacity_kWh,
            )
            - current_charge_kWh
        )
        current_charge_kWh += realised_pv_charge_amount
        actual_pv_charge_rate = kWh_to_kW(realised_pv_charge_amount)
        actual_pv_export = realised_pv_charge_amount - kW_to_kWh(pv_power)

        # charge the battery with grid next
        realised_grid_charge_rate = clamp_func(
            attempted_grid_charge_rate,
            -torch.tensor(self.max_charge_rate_kW).to(grid_action.device),
            self.max_charge_rate_kW - actual_pv_charge_rate,
        )
        provisional_grid_charge_amount = kW_to_kWh(realised_grid_charge_rate)
        actual_grid_charge_amount = (
            clamp_func(
                current_charge_kWh + provisional_grid_charge_amount,
                0,
                self.capacity_kWh,
            )
            - current_charge_kWh
        )
        current_charge_kWh += actual_grid_charge_amount
        if is_peak_time_if_taxed is not None:
            cost1 = self.round_or_not(
                taxman(actual_grid_charge_amount, is_peak_time_if_taxed[:, None], price)
            )
            cost2 = self.round_or_not(
                taxman(actual_pv_export, is_peak_time_if_taxed[:, None], price)
            )
            cost = cost1 + cost2
            # cost = taxman(
            #     self.round_or_not(
            #         (price / 1000) * (actual_grid_charge_amount + actual_pv_export)
            #     ),
            #     is_peak_time_if_taxed,
            # )
        else:
            cost = self.round_or_not(
                price / 1000 * (actual_grid_charge_amount + actual_pv_export)
            )
        if not output_debug_info:
            return current_charge_kWh, cost
        else:
            debug_info = {
                "actual_pv_charge_rate": actual_pv_charge_rate.squeeze().numpy().item(),
                "actual_pv_export": actual_pv_export.squeeze().numpy().item(),
                "actual_grid_charge_rate": realised_grid_charge_rate.squeeze()
                .numpy()
                .item(),
                "actual_grid_charge_amount": actual_grid_charge_amount.squeeze()
                .numpy()
                .item(),
                "attempted_pv_charge_rate": attempted_pv_charge_rate.squeeze()
                .numpy()
                .item(),
                "attempted_grid_charge_rate": attempted_grid_charge_rate.squeeze()
                .numpy()
                .item(),
                "realised_pv_charge_rate": realised_pv_charge_rate.squeeze()
                .numpy()
                .item(),
                "provisional_pv_charge_amount": provisional_pv_charge_amount.squeeze()
                .numpy()
                .item(),
                "provisional_grid_charge_amount": provisional_grid_charge_amount.squeeze()
                .numpy()
                .item(),
            }
            return current_charge_kWh, cost, debug_info


# in this version of battery env, there is no internal battery state, and we always process
# an entire batch of actions at once, computing their effect on the battery state over time, and find the cost / profit
class SeqBatteryEnv(nn.Module):
    def __init__(
        self,
        capacity_kWh: float,
        max_charge_rate_kW: float,
        initial_charge_kWh: float,
        round_profit: bool = False,
    ):
        super().__init__()
        self.capacity_kWh = capacity_kWh
        self.initial_charge_kWh = initial_charge_kWh
        self.max_charge_rate_kW = max_charge_rate_kW
        assert self.initial_charge_kWh <= self.capacity_kWh
        self.round_or_not = (
            partial(torch.round, decimals=4) if round_profit else lambda x: x
        )

    def get_neutral_trace(self, batch_size: int, time_steps: int) -> torch.Tensor:
        return torch.full((batch_size, time_steps + 1), self.initial_charge_kWh)

    def get_random_init_trace(self, batch_size: int, time_steps: int) -> torch.Tensor:
        trace = torch.full(
            (batch_size, time_steps + 1), torch.rand(batch_size) * self.capacity_kWh
        )
        return trace

    @staticmethod
    def soft_clamp_func(x, min_val, max_val, beta):
        return min_val + F.softplus(x - min_val, beta) - F.softplus(x - max_val, beta)

    def forward(
        self,
        grid_action: torch.Tensor,  # (batch_size, time_steps)
        pv_action: torch.Tensor,  # (batch_size, time_steps)
        pv_power: torch.Tensor,  # (batch_size, time_steps)
        price: torch.Tensor,  # (batch_size, time_steps)
        beta: float = None,
        random_initial_state: bool = False,
        is_peak_time_if_taxed: Optional[
            torch.Tensor
        ] = None,  # (batch_size, time_steps)
        output_debug_info: bool = False,
        input_raw_kWh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if beta is not None:
            clamp_func = partial(self.soft_clamp_func, beta=beta)
        else:
            clamp_func = torch.clamp

        time_steps = grid_action.size(1)
        batch_size = grid_action.size(0)
        # initialize battery state trace
        battery_state_trace = torch.zeros(batch_size, time_steps + 1).to(
            grid_action.device
        )
        if random_initial_state:
            battery_state_trace[:, 0] = (
                torch.rand(batch_size).to(grid_action.device) * self.capacity_kWh
            )
        else:
            battery_state_trace[:, 0] = self.initial_charge_kWh

        # process solar first
        attempted_pv_charge_rate = (
            pv_action * pv_power if not input_raw_kWh else pv_action
        )
        realised_pv_charge_rate = clamp_func(
            attempted_pv_charge_rate, 0, self.max_charge_rate_kW
        )
        provisional_pv_charge_amount = kW_to_kWh(realised_pv_charge_rate)

        # process grid next
        attempted_grid_charge_rate = (
            grid_action * self.max_charge_rate_kW if not input_raw_kWh else grid_action
        )
        # pre-allocation
        realised_pv_charge_amount = torch.zeros_like(grid_action).to(grid_action.device)
        actual_pv_charge_rate = torch.zeros_like(grid_action).to(grid_action.device)
        actual_pv_export = torch.zeros_like(grid_action).to(grid_action.device)
        realised_grid_charge_rate = torch.zeros_like(grid_action).to(grid_action.device)
        provisional_grid_charge_amount = torch.zeros_like(grid_action).to(
            grid_action.device
        )
        actual_grid_charge_amount = torch.zeros_like(grid_action).to(grid_action.device)

        for i in range(time_steps):
            # charge the battery with pv first
            after_pv_charge_state = clamp_func(
                battery_state_trace[:, i] + provisional_pv_charge_amount[:, i],
                0,
                self.capacity_kWh,
            )
            realised_pv_charge_amount[:, i] = (
                after_pv_charge_state - battery_state_trace[:, i]
            )
            actual_pv_charge_rate[:, i] = kWh_to_kW(realised_pv_charge_amount[:, i])
            actual_pv_export[:, i] = realised_pv_charge_amount[:, i] - kW_to_kWh(
                pv_power[:, i]
            )
            # charge the battery with grid next
            realised_grid_charge_rate[:, i] = clamp_func(
                attempted_grid_charge_rate[:, i],
                -self.max_charge_rate_kW * torch.ones_like(actual_pv_charge_rate[:, i]),
                self.max_charge_rate_kW - actual_pv_charge_rate[:, i],
            )
            provisional_grid_charge_amount[:, i] = kW_to_kWh(
                realised_grid_charge_rate[:, i]
            )
            battery_state_trace[:, i + 1] = clamp_func(
                after_pv_charge_state + provisional_grid_charge_amount[:, i],
                0,
                self.capacity_kWh,
            )
            actual_grid_charge_amount[:, i] = (
                battery_state_trace[:, i + 1] - after_pv_charge_state
            )

        # apply tariff based on peak time and flow direction
        if is_peak_time_if_taxed is not None:
            cost1 = self.round_or_not(
                taxman(actual_grid_charge_amount, is_peak_time_if_taxed, price)
            )
            cost2 = self.round_or_not(
                taxman(actual_pv_export, is_peak_time_if_taxed, price)
            )
            cost = cost1 + cost2
            # cost = taxman(
            #     self.round_or_not(
            #         (price / 1000) * (actual_grid_charge_amount + actual_pv_export)
            #     ),
            #     is_peak_time_if_taxed,
            # )
        else:
            cost = self.round_or_not(
                price / 1000 * (actual_grid_charge_amount + actual_pv_export)
            )

        if not output_debug_info:
            return battery_state_trace, cost
        else:
            debug_info = {
                "actual_pv_charge_rate": actual_pv_charge_rate,
                "actual_pv_export": actual_pv_export,
                "actual_grid_charge_rate": realised_grid_charge_rate,
                "actual_grid_charge_amount": actual_grid_charge_amount,
                "attempted_pv_charge_rate": attempted_pv_charge_rate,
                "attempted_grid_charge_rate": attempted_grid_charge_rate,
                "realised_pv_charge_rate": realised_pv_charge_rate,
                "provisional_pv_charge_amount": provisional_pv_charge_amount,
                "provisional_grid_charge_amount": provisional_grid_charge_amount,
            }
            return battery_state_trace, cost, debug_info

    def static_forward(
        self,
        battery_state_trace: torch.Tensor,
        grid_action: torch.Tensor,  # (batch_size, time_steps)
        pv_action: torch.Tensor,  # (batch_size, time_steps)
        pv_power: torch.Tensor,  # (batch_size, time_steps)
        price: torch.Tensor,  # (batch_size, time_steps)
        beta: float = None,
        is_peak_time_if_taxed: Optional[torch.Tensor] = None,
    ):
        """
        Apply the given action to the battery.
        This version of the method does not cumulatively update the battery state, but instead
        assumes that the battery state is provided as an input.
        """
        if beta is not None:
            clamp_func = partial(self.soft_clamp_func, beta=beta)
        else:
            clamp_func = torch.clamp
        # process solar first
        attempted_pv_charge_rate = pv_action * pv_power
        realised_pv_charge_rate = clamp_func(
            attempted_pv_charge_rate, 0, self.max_charge_rate_kW
        )
        provisional_pv_charge_amount = kW_to_kWh(realised_pv_charge_rate)

        # process grid next
        attempted_grid_charge_rate = grid_action * self.max_charge_rate_kW

        # in this function we ignore the cumulative effect of the battery state
        # charge the battery with pv first
        after_pv_charge_state = clamp_func(
            battery_state_trace + provisional_pv_charge_amount,
            0,
            self.capacity_kWh,
        )  # only for current time, battery state after pv charge
        realised_pv_charge_amount = after_pv_charge_state - battery_state_trace
        actual_pv_charge_rate = kWh_to_kW(realised_pv_charge_amount)
        actual_pv_export = realised_pv_charge_amount - kW_to_kWh(pv_power)

        realised_grid_charge_rate = clamp_func(
            attempted_grid_charge_rate,
            -self.max_charge_rate_kW * torch.ones_like(actual_pv_charge_rate),
            self.max_charge_rate_kW - actual_pv_charge_rate,
        )
        provisional_grid_charge_amount = kW_to_kWh(realised_grid_charge_rate)
        battery_state_trace = clamp_func(
            after_pv_charge_state + provisional_grid_charge_amount,
            0,
            self.capacity_kWh,
        )
        actual_grid_charge_amount = battery_state_trace - after_pv_charge_state

        if is_peak_time_if_taxed is not None:
            cost1 = self.round_or_not(
                taxman(actual_grid_charge_amount, is_peak_time_if_taxed, price)
            )
            cost2 = self.round_or_not(
                taxman(actual_pv_export, is_peak_time_if_taxed, price)
            )
            cost = cost1 + cost2
        else:
            cost = self.round_or_not(
                price / 1000 * (actual_grid_charge_amount + actual_pv_export)
            )
        return battery_state_trace, cost


def taxman(
    flow: torch.Tensor, is_peak_time: torch.Tensor, price: torch.Tensor
) -> torch.Tensor:
    original_cost = price / 1000 * flow
    return original_cost + torch.where(
        is_peak_time,
        torch.where(flow <= 0, -0.3, 0.4),
        torch.where(flow <= 0, 0.15, 0.05),
    ) * torch.abs(original_cost)


def get_peak_indicator(
    timestamp: pd.Series, peak_lower: int = 17, peak_upper: int = 21
) -> torch.Tensor:
    plus_10 = pd.Timedelta(hours=10)
    timestamp = timestamp + plus_10
    is_peak = (timestamp.dt.hour >= peak_lower) & (timestamp.dt.hour < peak_upper)
    return torch.tensor(is_peak.values).to(torch.bool)
