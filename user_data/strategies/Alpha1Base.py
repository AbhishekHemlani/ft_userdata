"""
The base template for all Pareto strategies in Freqtrade V3
"""
# --- Do not remove these libs ---
import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
import numpy as np
import pandas as pd
from pandas import DataFrame
pd.options.mode.chained_assignment = None
# --------------------------------

# --- Freqtrade & Related libs ---
from freqtrade.strategy import IStrategy
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

from freqtrade.persistence import Order, PairLocks, Trade


from freqtrade.exchange.exchange import Exchange

import talib.abstract as ta

import freqtrade.vendor.qtpylib.indicators as qtpylib

from technical.candles import heikinashi as heik
# --------------------------------

# __CONSTANTS__
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)




# __FUNCTIONS__



# __CLASSES__
class CustomPairDatabank(object):
    '''
    Store custom trading data in a central repo for each pair accessable to all callbacks in the strategy class.
    '''

    # trade_status values
    ENTRY = 'entry'
    EXIT = 'exit'
    NONE = None

    # PORTFOLIO DERISK
    IS_PORTFOLIO_DERISK_LONG = False
    IS_PORTFOLIO_DERISK_SHORT = False

    DEFAULT_DICT = {
        'retry': 0,  # 0/1 : Attempt retry of limit order execution on entries
        'retry_starttime': datetime.datetime.utcnow() - datetime.timedelta(minutes=int(60)),
        'trade_status': None,  # 'entry'/'exit'/None: track whether a pair is attempting entry or exit
        'retry_count': 0,  # int: num of times we have attempted reentry
        'side': 'long',  # 'long'/'short': direction of the entry signal
        'position_pnl': 0., #'float': pcnt pnl for the live trade at market prices
    }

    def __init__(self, MAX_ENTRY_RETRY_MINUTES, MAX_ENTRY_RETRY_COUNT, PORTFOLIO_DERISK_POSITIONS, PORTFOLIO_DERISK_PNL):
        # retry_starttime values
        self.MAX_ENTRY_RETRY_MINUTES = MAX_ENTRY_RETRY_MINUTES
        self.MAX_ENTRY_RETRY_COUNT = MAX_ENTRY_RETRY_COUNT

        # portfolio derisk settings
        self.PORTFOLIO_DERISK_POSITIONS = PORTFOLIO_DERISK_POSITIONS
        self.PORTFOLIO_DERISK_PNL = -abs(PORTFOLIO_DERISK_PNL)

        # List all pairs
        self.pairs = {}

    def check_pair(self, pair):
        if not (pair in self.pairs.keys()):
            self.pairs[pair] = self.DEFAULT_DICT.copy()

    def get_val(self, pair, dataType):
        self.check_pair(pair)
        return self.pairs[pair][dataType]

    def set_val(self, pair, dataType, val):
        self.pairs[pair][dataType] = val

    def update_portfolio_derisk(self) -> None:
        """Update the portfolio derisk boolean in-memory"""
        for side in ['long', 'short']:

            # Get long pairs and get short pairs
            pairs = {key: val for key, val in self.pairs.items() if val['side'] == side}
            current_derisk_value = (self.IS_PORTFOLIO_DERISK_LONG if side == 'long' else self.IS_PORTFOLIO_DERISK_SHORT)

            # Extract portfolio underwater values
            PositionPnl = [pair_details['position_pnl'] for pair, pair_details in pairs.items()]
            NumUnderwaterPositions = int(sum([1 if pnl <= self.PORTFOLIO_DERISK_PNL else 0 for pnl in PositionPnl]))
            NumPositions = int(sum([1 if pnl != 0. else 0 for pnl in PositionPnl])) # pnl===0 indicats an exited position

            # Set the portfolio derisk value...
            # ... The entire portfolio is underwater below our hurdle value, so START derisking
            if (NumUnderwaterPositions == NumPositions) and (NumUnderwaterPositions > self.PORTFOLIO_DERISK_POSITIONS):
                if side == 'long':
                    self.IS_PORTFOLIO_DERISK_LONG = True
                else:
                    self.IS_PORTFOLIO_DERISK_SHORT = True

            # ... NO positions are active BUT we are already derisking, so STOP derisking
            elif (NumPositions == 0) and current_derisk_value:
                if side == 'long':
                    self.IS_PORTFOLIO_DERISK_LONG = False
                else:
                    self.IS_PORTFOLIO_DERISK_SHORT = False

            # ... Do not adjust the boolean value in any other sceneario



class ParetoStrategyBase(IStrategy):
    """
    Base Pareto Strategy Template (SPOT & FUTURE)
    """

    # SET TO A 15m TRADING TIMEFRAME!!!
    TIMESHIFT = {
        '4H': int(4 * 4),
        '1H': int(4),
        '30T': 2,
        '15T': 1,
    }

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Base Leverage
    LEVERAGE_TARGET = 1.0

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.15

    # Optimal timeframe for the strategy
    timeframe = '4h'

    # run "populate_indicators" only for new candle
    #process_only_new_candles = True
    #ignore_buying_expired_candle_after = 20 * 60 # (seconds)

    # Custom Databank
    CUST_SHORT_TAG = 'force_enter_short'
    CUST_LONG_TAG = 'force_enter_long'
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )
    CONSECUTIVE_ENTRY_CONFIRMATIONS = 6

    # Dynamic Stoploss/Exiting
    use_custom_stoploss = True
    trailing_stop_positive = 0.04
    TIMEFRAMES = ['4H', '1H', '30T', '15T']
    PCNT_VOLUME_HURDLE = 0.

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        # Extract Linear Regressions
        dataframe['linreg'] = ta.TSF(dataframe['close'], timeperiod=min(960, len(dataframe)))
        dataframe['linreg_slope'] = 10000. * ta.LINEARREG_SLOPE(dataframe['close'], timeperiod=min(960, len(dataframe))) / dataframe['close']

        # // Multiple Timeseries Calculations //
        for timeframe in self.TIMEFRAMES:

            # Set the time shift
            time_shift = self.TIMESHIFT[timeframe]

            # Rolling
            dataframe[f"{timeframe}_volume"] = dataframe['volume'].rolling(time_shift).sum()

            # Technicals
            dataframe[f"{timeframe}_di_plus"] = ta.PLUS_DI(dataframe, timeperiod=6 * time_shift)
            dataframe[f"{timeframe}_di_minus"] = ta.MINUS_DI(dataframe, timeperiod=6 * time_shift)
            dataframe[f"{timeframe}_rsi"] = ta.RSI(dataframe, timeperiod=14 * time_shift)
            dataframe[f"{timeframe}_sar"] = ta.SAR(dataframe, acceleration=2. / (6 * time_shift))

            # Heikin Ashi
            dfRolling = pd.DataFrame()
            dfRolling['open'] = dataframe['open'].shift(time_shift)
            dfRolling['high'] = dataframe['high'].rolling(time_shift).max()
            dfRolling['low'] = dataframe['low'].rolling(time_shift).min()
            dfRolling['close'] = dataframe['close']
            dfRolling.bfill(axis='rows', inplace=True) #heik needs to start on a valid price value, so backfill empty lagged prices

            heikinashi = heik(dfRolling)
            dataframe[f'{timeframe}_ha_open'] = heikinashi['open']
            dataframe[f'{timeframe}_ha_close'] = heikinashi['close']
            dataframe[f'{timeframe}_ha_high'] = heikinashi['high']
            dataframe[f'{timeframe}_ha_low'] = heikinashi['low']

        # //ENTRY & EXIT BOOLEANS//
        TF = "4H"
        SHIFT = self.TIMESHIFT[TF]

        # <ENTER LONG>
        enter_long = []
        # ... DMI
        enter_long.append((dataframe[f'{TF}_di_plus'] > dataframe[f'{TF}_di_minus']))
        # ... RSI
        enter_long.append((dataframe[f'{TF}_rsi'] < 75))
        # ... SAR
        enter_long.append((dataframe[f'{TF}_sar'] < dataframe['open']))
        # ... Renko
        enter_long.append((dataframe[f'{TF}_ha_close'] > dataframe[f'{TF}_ha_open'].shift(1 * SHIFT)))
        enter_long.append((dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) < dataframe[
            f'{TF}_ha_open'].shift(2 * SHIFT)))
        # ... Linear Regression Channel
        enter_long.append((dataframe[f'{TF}_ha_close'] < dataframe[f'linreg']))
        enter_long.append((dataframe[f'linreg_slope'] > 0.))
        dataframe.loc[
            (reduce(lambda x, y: x & y, enter_long)),
            ['enter_long', 'enter_tag']
        ] = (1, "alpha1_long")

        # <EXIT LONG>
        exit_long = []
        # ... Renko Reversal
        exit_long.append(
            (dataframe[f'{TF}_ha_close'] < dataframe[f'{TF}_ha_open'].shift(1*SHIFT)) \
            & (dataframe[f'{TF}_ha_close'].shift(1*SHIFT) > dataframe[f'{TF}_ha_open'].shift(2*SHIFT))
        )
        # ... Linear Regression Channel
        exit_long.append(
            (dataframe[f'{TF}_ha_close'] < dataframe[f'linreg']) \
            & (dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) > dataframe[f'linreg'].shift(1 * SHIFT))
        )
        exit_long.append(
            (dataframe[f'linreg_slope'] < 0) \
            & (dataframe[f'linreg_slope'].shift(1 * SHIFT) > 0)
        )

        dataframe.loc[
            reduce(lambda x, y: x | y, exit_long),
            ['exit_long', 'exit_tag']] = (1, 'strategy_exit_long')

        # //ENTER SHORT//
        enter_short = []
        # ... DMI
        enter_short.append((dataframe[f'{TF}_di_plus'] < dataframe[f'{TF}_di_minus']))
        # ... RSI
        enter_short.append((dataframe[f'{TF}_rsi'] > 25))
        # ... SAR
        enter_short.append((dataframe[f'{TF}_sar'] > dataframe['open']))
        # ... Renko
        enter_short.append((dataframe[f'{TF}_ha_close'] < dataframe[f'{TF}_ha_open'].shift(1 * SHIFT)))
        enter_short.append((dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) > dataframe[
            f'{TF}_ha_open'].shift(2 * SHIFT)))
        # ... Linear Regression Channel
        enter_short.append((dataframe[f'{TF}_ha_close'] > dataframe[f'linreg']))
        enter_short.append((dataframe[f'linreg_slope'] < 0.))
        dataframe.loc[
            (reduce(lambda x, y: x & y, enter_short)),
            ['enter_short', 'enter_tag']
        ] = (1, "alpha1_short")

        # Exit Short Position Logic
        exit_short = []
        # ... Renko Reversal
        exit_short.append(
            (dataframe[f'{TF}_ha_close'] > dataframe[f'{TF}_ha_open'].shift(1*SHIFT)) \
            & (dataframe[f'{TF}_ha_close'].shift(1*SHIFT) < dataframe[f'{TF}_ha_open'].shift(2*SHIFT))
        )
        # ... Linear Regression Channel
        exit_short.append(
            (dataframe[f'{TF}_ha_close'] > dataframe[f'linreg']) \
            & (dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) < dataframe[f'linreg'].shift(1 * SHIFT))
        )
        exit_short.append(
            (dataframe[f'linreg_slope'] > 0) \
            & (dataframe[f'linreg_slope'].shift(1 * SHIFT) < 0)
        )

        dataframe.loc[
            reduce(lambda x, y: x | y, exit_short),
            ['exit_short', 'exit_tag']] = (1, 'strategy_exit_short')

        # //Return the full dataframe//
        return dataframe

    def reentry_populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Custom version of enter_trend with an auto-retry feature for limits on LONGS OR SHORTS...
        Only attempt to enter after we place a force-enter trade. We track this info in cust_data object
        and we check that the retry attempt is valid here every minute.
        """

        current_pair = metadata['pair']  # QUOTE/BASE market pairing

        # If 0 < count < MAX_COUNT, we can attempt a retry. Else, no retry
        retry_count = self.cust_data.get_val(current_pair, 'retry_count')
        retry_timestamp = self.cust_data.get_val(current_pair, 'retry_starttime')
        retry_duration = (datetime.datetime.utcnow() - retry_timestamp).total_seconds() / 60.
        retry_side = self.cust_data.get_val(current_pair, 'side')

        is_retry_attempt = (0 < retry_count)
        not_exceeded_retries = (retry_count < self.cust_data.MAX_ENTRY_RETRY_COUNT)
        is_duration_small = (retry_duration < self.cust_data.MAX_ENTRY_RETRY_MINUTES)
        if is_retry_attempt and not_exceeded_retries and is_duration_small:
            self.cust_data.set_val(current_pair, 'retry', 1)
        else:
            # In all other scenarios, no attempt should be made
            self.cust_data.set_val(current_pair, 'retry', 0)

        # Populate the signal dataframe
        if (self.cust_data.get_val(current_pair, 'retry') == 1):
            if (retry_side == 'long'):
                dataframe[['enter_long', 'enter_tag']] = (1, 'reetnry_attempt')
            elif (retry_side == 'sell'):
                dataframe[['enter_short', 'enter_tag']] = (1, 'reetnry_attempt')

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Final callback to populate entry signals (long & short)
        """
        # Clean entry and exit signals
        dataframe.fillna(0, inplace=True)

        # The base entry (long/short) logic with re-entry attempts
        df = self.reentry_populate_entry_trend(dataframe, metadata)

        # Only enter once we receive multiple confirmations
        long_entry_confirmations = np.where(
            np.cumsum(df['enter_long']) >= self.CONSECUTIVE_ENTRY_CONFIRMATIONS,
            1.,
            0.,
        )
        short_entry_confirmations = np.where(
            np.cumsum(df['enter_short']) >= self.CONSECUTIVE_ENTRY_CONFIRMATIONS,
            1.,
            0.,
        )
        df['enter_long'] = long_entry_confirmations
        df['enter_short'] = short_entry_confirmations

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Final callback to populate exit signals (long & short)
        """
        # Clean entry and exit signals
        dataframe.fillna(0, inplace=True)

        #

        # Properly net exit signals with counterparty entry signals
        dataframe['exit_long'] = np.minimum(dataframe['exit_long'] + dataframe['enter_short'], 1)
        dataframe['exit_short'] = np.minimum(dataframe['exit_short'] + dataframe['enter_long'], 1)
        return dataframe

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        # HARD RESET the cust_data pnl value once we exit a position
        self.cust_data.set_val(pair, 'position_pnl', 0.)

        # HARD RESET of reentry bot to ensure we unlock force-entries
        self.cust_data.set_val(pair, 'retry_count', 0)
        self.cust_data.set_val(pair, 'retry_starttime', datetime.datetime.utcnow())
        return True

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        # HARD CHECK that the portfolio derisk is no longer in effect
        self.cust_data.update_portfolio_derisk()
        if (
                side == 'long' and self.cust_data.IS_PORTFOLIO_DERISK_LONG
        ) or (
                side == 'short' and self.cust_data.IS_PORTFOLIO_DERISK_SHORT
        ):
            return False

        # HARD CHECK to ensure we are not retrying too much.
        # :NOTE: This logic will halt force entries in the scenario where we do not update populate_entry_trend
        retry_count = self.cust_data.get_val(pair, 'retry_count')
        retry_timestamp = self.cust_data.get_val(pair, 'retry_starttime')
        retry_duration = (datetime.datetime.utcnow() - retry_timestamp).total_seconds() / 60.
        # This logic forced the bot to re-enter after waiting a while so no edge cases stop us from trading
        if retry_duration >= 2.0 * self.cust_data.MAX_ENTRY_RETRY_MINUTES:
            retry_count = 0
        if (retry_count == 0):
            # THIS IS THE FIRST FORCE_ENTRY ATTEMPT
            self.cust_data.set_val(pair, 'retry_count', retry_count + 1)
            self.cust_data.set_val(pair, 'retry_starttime', datetime.datetime.utcnow())
            self.cust_data.set_val(pair, 'side', side)
            return True
        elif (0 < retry_count) and (retry_count < self.cust_data.MAX_ENTRY_RETRY_COUNT) and (retry_duration < self.cust_data.MAX_ENTRY_RETRY_MINUTES):
            # THIS IS A RETRY, WE CAN KEEP IT GOING
            self.cust_data.set_val(pair, 'retry_count', retry_count + 1)
            return True
        elif (retry_count >= self.cust_data.MAX_ENTRY_RETRY_COUNT) or (retry_duration > self.cust_data.MAX_ENTRY_RETRY_MINUTES):
            # NO LONGER A VALID RETRY, PLEASE STOP PLACING ORDERS
            self.cust_data.set_val(pair, 'retry_count', 0)
            return False
        else:
            # SOMETHING IS WRONG, KEEP WAITING UNTIL DURATION PASSES NO TRADE OCCURS
            self.cust_data.set_val(pair, 'retry_count', retry_count + 1)
            return False

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Custom exit signal logic indicating that specified position should be sold. Returning a
        string or True from this method is equal to setting exit signal on a candle at specified
        time. This method is not called when exit signal is set.

        This method should be overridden to create exit signals that depend on trade parameters. For
        example you could implement an exit relative to the candle when the trade was opened,
        or a custom 1:2 risk-reward ROI.

        Custom exit reason max length is 64. Exceeding characters will be removed.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return: To execute exit, return a string with custom exit reason or True. Otherwise return
        None or False.
        """
        # // Update the cust_data object with position PnL//

        self.cust_data.set_val(pair, 'position_pnl', current_profit) # profit is pcnt!
        # // Ask cust_data if we are ready to de-risk the whole portfolio //
        self.cust_data.update_portfolio_derisk()
        # // The bot can continue trading once all positions are confirmed exited //
        if (
                (self.cust_data.get_val(pair, 'side') == 'long') and self.cust_data.IS_PORTFOLIO_DERISK_LONG
        ) or (
                (self.cust_data.get_val(pair, 'side') == 'short') and self.cust_data.IS_PORTFOLIO_DERISK_SHORT
        ):
            logger.info(f"CUSTOM EXIT | Trade Pair {pair} | Profit {current_profit}")
            return "portfolio_derisk_event"
        return None

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in ask_strategy.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the currentrate
        """
        # // Data Cache //
        # (!Please organize from slowest to fastest; makes later boolean conditions easier to calculate!)
        CACHE = {
            '4H': {
                'vol_spike_bool': False,
                'renko_rising_bool': False,
            },
            '1H': {
                'vol_spike_bool': False,
                'renko_rising_bool': False,
            },
            '30T': {
                'vol_spike_bool': False,
                'renko_rising_bool': False,
            },
            '15T': {
                'vol_spike_bool': False,
                'renko_rising_bool': False,
            },
        }

        def getCandle(shift_len: int = 0):
            """subroutine to call the squeezed candle data as a simple 1D dictionary, shifted in time if needed"""
            if not df.empty:
                return df.shift(shift_len, axis=0).iloc[-1].squeeze()
            return {
                f'{timeframe}_volume': 0.,
                f'{timeframe}_ha_close': 0.,
                f'{timeframe}_ha_open': 0.,
            }

        # Get the most recently updated dataframe with indicators (NO SHIFTING)
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # // Advanced Renko Reversal //
        # ... set the boolean conditions for each timeframe in CACHE
        for timeframe in CACHE:
            # Get the candle data
            candle0 = getCandle(0) # Most recent candle
            candle1 = getCandle(self.TIMESHIFT[timeframe]) # Candle Shifted back 1x timedelta
            candle2 = getCandle(2 * self.TIMESHIFT[timeframe]) # Candle Shifted back 2x timedelta

            # Clean any NaN values in the dictionaries

            # Calculate booleans to trigger repricing
            # ... volume spikes over the lookback
            CACHE[timeframe]["vol_spike_bool"] = (
                    candle0[f'{timeframe}_volume'] >= (1. + self.PCNT_VOLUME_HURDLE) * candle1[f'{timeframe}_volume']
            )

            # ... renko signals are rising in the direction of the trade
            if trade.is_short:
                CACHE[timeframe]["renko_rising_bool"] = (
                    (candle0[f'{timeframe}_ha_close'] <= candle1[f'{timeframe}_ha_open'])
                ) and (
                    (candle1[f'{timeframe}_ha_close'] >= candle2[f'{timeframe}_ha_open'])
                )
            else:
                CACHE[timeframe]["renko_rising_bool"] = (
                    (candle0[f'{timeframe}_ha_close'] >= candle1[f'{timeframe}_ha_open'])
                ) and (
                    (candle1[f'{timeframe}_ha_close'] <= candle2[f'{timeframe}_ha_open'])
                )

        # For each timeframe, determine if we meet the stoploss reset condition: rising volume and rising
        # ... renko candles for all timeframes (t < T)
        stoploss_conditions = []
        all_timeframes = list(CACHE.keys())
        i = 0
        while(i<len(all_timeframes)):
            j = 0
            all_prior_bools = []
            while(j <= i):
                timeframe = all_timeframes[j]
                all_prior_bools.append(
                    CACHE[timeframe]['vol_spike_bool'] and CACHE[timeframe]['renko_rising_bool']
                )
                j += 1
            stoploss_conditions.append(np.all(all_prior_bools))
            i += 1

        # Determine if we should reset stoploss
        if np.all(stoploss_conditions):
            return -self.trailing_stop_positive

        # Return maximum stoploss value, keeping current stoploss price unchanged
        return -1



class USDStrategy001(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False



class USDStrategy003(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.20
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False



class BTCStrategy001(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False



class BTCStrategy003(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False



class ETHStrategy001(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Allow For Retry Attempts
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=6,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.02
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe


class ETHStrategy003(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Allow For Retry Attempts
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe


class USDComboStrategy1(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "60":  0.10,
        "30":  0.10,
        "20":  0.10,
        "0":  0.10
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False



class USDComboStrategy2(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.20

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False



class BTCComboStrategy1(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "60": 0.10,
        "30": 0.10,
        "20": 0.10,
        "0": 0.10
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False



class BTCComboStrategy2(ParetoStrategyBase):
    TIMESHIFT = {
        '4H': int(48),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.20

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False



class ETHComboStrategy1(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "60":  0.10,
        "30":  0.10,
        "20":  0.10,
        "0":  0.10
    }

    # Allow For Retry Attempts
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe



class ETHComboStrategy2(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Allow For Retry Attempts
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.20

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe


class BinanceUSDStrategy001(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = True

    TIMESHIFT = {
        '4H': int(12 * 4),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe




class BinanceUSDStrategy003(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = True

    TIMESHIFT = {
        '4H': int(12 * 4),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=0,
        MAX_ENTRY_RETRY_COUNT=0,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False




class BinanceUSDComboStrategy1(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = True

    TIMESHIFT = {
        '4H': int(12 * 4),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe



class BinanceUSDComboStrategy2(ParetoStrategyBase):
    # Can this strategy go short?
    can_short: bool = True

    TIMESHIFT = {
        '4H': int(12 * 4),
        '1H': int(12),
        '30T': 6,
        '15T': 3,
    }

    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=4,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    # Minimal ROI
    minimal_roi = {
        "0": 0.25, "240": 0.2146, "480": 0.1842, "720": 0.1581,
        "960": 0.1357, "1200": 0.1165, "1440": 0.1000, "1680": 0.0858,
        "1920": 0.0737, "2160": 0.0632, "2400": 0.0543, "2640": 0.0466,
        "2880": 0.0400,
    }

    # Trailing Stoploss
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.2
    trailing_only_offset_is_reached = False

    # Experimental settings (config will overide these if set)
    use_exit_signal = False
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = ParetoStrategyBase.populate_entry_trend(self, dataframe, metadata)
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

# freqtrade trade -s prodBinanceUSDStrategy003 -c ./user_data/config-sandbox-binance.json -c ./user_data/config-sandbox-binance-private.json