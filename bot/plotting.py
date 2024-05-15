import matplotlib.pyplot as plt

def plot_results(profits, market_prices, battery_soc, actions):
    """
    Plot the bids, profits, market prices, and battery state of charge over time.

    :param bids: List of bid prices for each time step.
    :param profits: List of profits for each time step.
    :param market_prices: List of market prices for each time step.
    :param battery_soc: List of battery state of charge values for each time step.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot bids and market prices on the first axis
    ax1.plot(market_prices, label='Market Price', color='green', linestyle='--')
    ax1.set_ylabel('Price ($/kWh)', color='blue')
    ax1.tick_params('y', colors='blue')
    ax1.legend(loc='upper left')

    # Create a second axis for profits on the first subplot
    ax1_profit = ax1.twinx()
    ax1_profit.plot(profits, label='Profit', color='red')
    ax1_profit.set_ylabel('Profit ($)', color='red')
    ax1_profit.tick_params('y', colors='red')
    ax1_profit.legend(loc='upper right')

    # Plot battery state of charge on the second subplot
    ax2.plot(battery_soc, label='Battery State of Charge', color='purple')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Battery State of Charge (kWh)', color='purple')
    ax2.tick_params('y', colors='purple')
    ax2.legend(loc='upper left')

    # plot the actions (buy/sell amount)
    # plot with very thin transparent line
    ax3.plot(actions, label='Actions', color='blue', alpha=0.1, linewidth=0.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Actions (kWh)', color='blue')
    ax3.tick_params('y', colors='blue')
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()