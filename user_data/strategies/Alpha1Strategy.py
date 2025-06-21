from freqtrade.strategy import IStrategy
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from freqtrade.persistence import Trade
from pandas import DataFrame
import datetime
import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
from technical.candles import heikinashi as heik

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None

# Timeframe shift constants
TIMESHIFT = {
    '4H': 48,
    '1H': 12,
    '30T': 6, 
    '15T': 3,
}

class CustomPairDatabank:
    ENTRY = 'entry'
    EXIT = 'exit'
    NONE = None
    IS_PORTFOLIO_DERISK_LONG = False
    IS_PORTFOLIO_DERISK_SHORT = False
    DEFAULT_DICT = {
        'retry': 0,
        'retry_starttime': datetime.datetime.utcnow() - datetime.timedelta(minutes=60),
        'trade_status': None,
        'retry_count': 0,
        'side': 'long',
        'position_pnl': 0.
    }

    def __init__(self, MAX_ENTRY_RETRY_MINUTES, MAX_ENTRY_RETRY_COUNT, PORTFOLIO_DERISK_POSITIONS, PORTFOLIO_DERISK_PNL):
        self.MAX_ENTRY_RETRY_MINUTES = MAX_ENTRY_RETRY_MINUTES
        self.MAX_ENTRY_RETRY_COUNT = MAX_ENTRY_RETRY_COUNT
        self.PORTFOLIO_DERISK_POSITIONS = PORTFOLIO_DERISK_POSITIONS
        self.PORTFOLIO_DERISK_PNL = -abs(PORTFOLIO_DERISK_PNL)
        self.pairs = {}
        logger.info(f"CustomPairDatabank initialized - Max retry: {MAX_ENTRY_RETRY_COUNT}, Portfolio derisk positions: {PORTFOLIO_DERISK_POSITIONS}, PNL threshold: {PORTFOLIO_DERISK_PNL}")

    def check_pair(self, pair):
        if pair not in self.pairs:
            self.pairs[pair] = self.DEFAULT_DICT.copy()
            logger.debug(f"Initialized new pair data for {pair}")

    def get_val(self, pair, dataType):
        self.check_pair(pair)
        return self.pairs[pair][dataType]

    def set_val(self, pair, dataType, val):
        self.check_pair(pair)
        old_val = self.pairs[pair][dataType]
        self.pairs[pair][dataType] = val
        logger.debug(f"Updated {pair} {dataType}: {old_val} -> {val}")

    def update_portfolio_derisk(self):
        for side in ['long', 'short']:
            pairs = {key: val for key, val in self.pairs.items() if val['side'] == side}
            PositionPnl = [p['position_pnl'] for p in pairs.values()]
            # Only consider positions that have actually been evaluated (not new trades with pnl=0)
            ActivePositions = [pnl for pnl in PositionPnl if pnl != 0.]
            NumUnderwater = sum(1 for pnl in ActivePositions if pnl <= self.PORTFOLIO_DERISK_PNL)
            NumPositions = len(ActivePositions)

            if NumUnderwater == NumPositions and NumUnderwater > self.PORTFOLIO_DERISK_POSITIONS:
                if side == 'long':
                    if not self.IS_PORTFOLIO_DERISK_LONG:
                        logger.warning(f"PORTFOLIO DERISK ACTIVATED for {side} positions - {NumUnderwater}/{NumPositions} positions underwater")
                    self.IS_PORTFOLIO_DERISK_LONG = True
                else:
                    if not self.IS_PORTFOLIO_DERISK_SHORT:
                        logger.warning(f"PORTFOLIO DERISK ACTIVATED for {side} positions - {NumUnderwater}/{NumPositions} positions underwater")
                    self.IS_PORTFOLIO_DERISK_SHORT = True
            elif NumPositions == 0:
                if side == 'long' and self.IS_PORTFOLIO_DERISK_LONG:
                    logger.info(f"PORTFOLIO DERISK DEACTIVATED for {side} positions - no active positions")
                    self.IS_PORTFOLIO_DERISK_LONG = False
                elif side == 'short' and self.IS_PORTFOLIO_DERISK_SHORT:
                    logger.info(f"PORTFOLIO DERISK DEACTIVATED for {side} positions - no active positions")
                    self.IS_PORTFOLIO_DERISK_SHORT = False

class ParetoStrategyBase(IStrategy):
    INTERFACE_VERSION = 3
    LEVERAGE_TARGET = 1.0
    stoploss = -0.15
    timeframe = '4h'
    use_custom_stoploss = True
    trailing_stop_positive = 0.04
    TIMEFRAMES = ['4H', '1H', '30T', '15T']
    PCNT_VOLUME_HURDLE = 0.

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        raise NotImplementedError()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['enter_long'] == 1, 'enter_long'] = 1
        return dataframe

    def reentry_populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        retry_count = self.cust_data.get_val(pair, 'retry_count')
        retry_timestamp = self.cust_data.get_val(pair, 'retry_starttime')
        retry_duration = (datetime.datetime.utcnow() - retry_timestamp).total_seconds() / 60.
        retry_side = self.cust_data.get_val(pair, 'side')

        logger.debug(f"Reentry check for {pair}: count={retry_count}, duration={retry_duration:.2f}min, side={retry_side}")

        if 0 < retry_count < self.cust_data.MAX_ENTRY_RETRY_COUNT and retry_duration < self.cust_data.MAX_ENTRY_RETRY_MINUTES:
            self.cust_data.set_val(pair, 'retry', 1)
            logger.info(f"Reentry attempt approved for {pair} - attempt {retry_count}/{self.cust_data.MAX_ENTRY_RETRY_COUNT}")
        else:
            self.cust_data.set_val(pair, 'retry', 0)
            if retry_count > 0:
                logger.debug(f"Reentry attempt rejected for {pair} - count limit or time exceeded")

        if self.cust_data.get_val(pair, 'retry') == 1:
            if retry_side == 'long':
                dataframe[['enter_long', 'enter_tag']] = (1, 'reentry_attempt')
                logger.info(f"Reentry signal generated for {pair} - LONG")
            elif retry_side == 'sell':
                dataframe[['enter_short', 'enter_tag']] = (1, 'reentry_attempt')
                logger.info(f"Reentry signal generated for {pair} - SHORT")
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        self.cust_data.set_val(pair, 'position_pnl', current_profit)
        self.cust_data.update_portfolio_derisk()
        
        logger.debug(f"Custom exit check for {pair}: profit={current_profit:.4f}, side={self.cust_data.get_val(pair, 'side')}")
        
        if (
            self.cust_data.get_val(pair, 'side') == 'long' and self.cust_data.IS_PORTFOLIO_DERISK_LONG
        ) or (
            self.cust_data.get_val(pair, 'side') == 'short' and self.cust_data.IS_PORTFOLIO_DERISK_SHORT
        ):
            logger.warning(f"PORTFOLIO DERISK EXIT triggered for {pair} | Profit: {current_profit:.4f} | Side: {self.cust_data.get_val(pair, 'side')}")
            return "portfolio_derisk_event"
        return None

class Alpha1Strategy(ParetoStrategyBase):
    can_short = False
    cust_data = CustomPairDatabank(
        MAX_ENTRY_RETRY_MINUTES=60,
        MAX_ENTRY_RETRY_COUNT=10,
        PORTFOLIO_DERISK_POSITIONS=8,
        PORTFOLIO_DERISK_PNL=-0.04
    )

    print("ALPHA1 RUN")
    logger.info("Alpha1Strategy initialized")

    minimal_roi = {
        "0": 0.3,
        "1440": 0.15,
        "2880": 0.075,
        "4320": 0.0375,
        "5760": 0.01875,
        "7200": 0.009
    }

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        logger.debug(f"Calculating indicators for {pair} - dataframe length: {len(dataframe)}")
        
        for tf in self.TIMEFRAMES:
            shift = TIMESHIFT[tf]
            dataframe[f"{tf}_volume"] = dataframe['volume'].rolling(shift).sum()
            dataframe[f"{tf}_di_plus"] = ta.PLUS_DI(dataframe, timeperiod=6 * shift)
            dataframe[f"{tf}_di_minus"] = ta.MINUS_DI(dataframe, timeperiod=6 * shift)
            dataframe[f"{tf}_rsi"] = ta.RSI(dataframe, timeperiod=56 * shift)
            dataframe[f"{tf}_sar"] = ta.SAR(dataframe, acceleration=2. / (6 * shift))
            dataframe[f"{tf}_linreg"] = ta.TSF(dataframe, timeperiod=min(960, len(dataframe)))
            dataframe[f"{tf}_linreg_slope"] = 10000. * ta.LINEARREG_SLOPE(dataframe, timeperiod=min(960, len(dataframe))) / dataframe['open']

            dfRolling = pd.DataFrame()
            dfRolling['open'] = dataframe['open'].shift(shift)
            dfRolling['high'] = dataframe['high'].rolling(shift).max()
            dfRolling['low'] = dataframe['low'].rolling(shift).min()
            dfRolling['close'] = dataframe['close']
            dfRolling.bfill(inplace=True)
            ha = heik(dfRolling)
            dataframe[f'{tf}_ha_open'] = ha['open']
            dataframe[f'{tf}_ha_close'] = ha['close']
            dataframe[f'{tf}_ha_high'] = ha['high']
            dataframe[f'{tf}_ha_low'] = ha['low']

        TF = "4H"
        SHIFT = TIMESHIFT[TF]

        dataframe['enter_long'] = (
            (dataframe[f'{TF}_di_plus'] > dataframe[f'{TF}_di_minus']) &
            (dataframe[f'{TF}_rsi'] < 75) &
            (dataframe[f'{TF}_sar'] < dataframe['open']) &
            (dataframe[f'{TF}_ha_close'] > dataframe[f'{TF}_ha_open'].shift(1 * SHIFT)) &
            (dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) < dataframe[f'{TF}_ha_open'].shift(2 * SHIFT))
        ).astype(int)

        # Add entry tag for proper identification
        dataframe.loc[dataframe['enter_long'] == 1, 'enter_tag'] = 'alpha1_long'

        dataframe['exit_long'] = (
            (dataframe[f'{TF}_ha_close'] < dataframe[f'{TF}_ha_open'].shift(1 * SHIFT)) &
            (dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) > dataframe[f'{TF}_ha_open'].shift(2 * SHIFT))
        ).astype(int)

        dataframe.fillna(0, inplace=True)
        logger.info(f"{metadata['pair']} - enter_long signals: {dataframe['enter_long'].sum()}")
        logger.info(f"{metadata['pair']} entries: {dataframe['enter_long'].sum()}, exits: {dataframe['exit_long'].sum()}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']
        exit_signals = dataframe['exit_long'].sum()
        logger.debug(f"{pair} - Processing exit trend, total exit signals: {exit_signals}")
        
        dataframe.loc[dataframe['exit_long'] == 1, 'exit_long'] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """Log trade entry confirmation"""
        logger.info(f"TRADE ENTRY CONFIRMED - Pair: {pair}, Side: {side}, Amount: {amount:.6f}, Rate: {rate:.6f}, Tag: {entry_tag}")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """Log trade exit confirmation"""
        profit = trade.calc_profit_ratio(rate)
        logger.info(f"TRADE EXIT CONFIRMED - Pair: {pair}, Profit: {profit:.4f}, Reason: {exit_reason}, Amount: {amount:.6f}, Rate: {rate:.6f}")
        return True

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Log custom stoploss calculations"""
        logger.debug(f"Custom stoploss check for {pair} - Current profit: {current_profit:.4f}, Rate: {current_rate:.6f}")
        return -0.15  # Default stoploss
