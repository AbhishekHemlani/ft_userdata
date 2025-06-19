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

    def check_pair(self, pair):
        if pair not in self.pairs:
            self.pairs[pair] = self.DEFAULT_DICT.copy()

    def get_val(self, pair, dataType):
        self.check_pair(pair)
        return self.pairs[pair][dataType]

    def set_val(self, pair, dataType, val):
        self.pairs[pair][dataType] = val

    def update_portfolio_derisk(self):
        for side in ['long', 'short']:
            pairs = {key: val for key, val in self.pairs.items() if val['side'] == side}
            PositionPnl = [p['position_pnl'] for p in pairs.values()]
            NumUnderwater = sum(1 for pnl in PositionPnl if pnl <= self.PORTFOLIO_DERISK_PNL)
            NumPositions = sum(1 for pnl in PositionPnl if pnl != 0.)

            if NumUnderwater == NumPositions and NumUnderwater > self.PORTFOLIO_DERISK_POSITIONS:
                if side == 'long':
                    self.IS_PORTFOLIO_DERISK_LONG = True
                else:
                    self.IS_PORTFOLIO_DERISK_SHORT = True
            elif NumPositions == 0:
                if side == 'long':
                    self.IS_PORTFOLIO_DERISK_LONG = False
                else:
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

        if 0 < retry_count < self.cust_data.MAX_ENTRY_RETRY_COUNT and retry_duration < self.cust_data.MAX_ENTRY_RETRY_MINUTES:
            self.cust_data.set_val(pair, 'retry', 1)
        else:
            self.cust_data.set_val(pair, 'retry', 0)

        if self.cust_data.get_val(pair, 'retry') == 1:
            if retry_side == 'long':
                dataframe[['enter_long', 'enter_tag']] = (1, 'reentry_attempt')
            elif retry_side == 'sell':
                dataframe[['enter_short', 'enter_tag']] = (1, 'reentry_attempt')
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        self.cust_data.set_val(pair, 'position_pnl', current_profit)
        self.cust_data.update_portfolio_derisk()
        if (
            self.cust_data.get_val(pair, 'side') == 'long' and self.cust_data.IS_PORTFOLIO_DERISK_LONG
        ) or (
            self.cust_data.get_val(pair, 'side') == 'short' and self.cust_data.IS_PORTFOLIO_DERISK_SHORT
        ):
            logger.info(f"CUSTOM EXIT | Trade Pair {pair} | Profit {current_profit}")
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

        dataframe['exit_long'] = (
            (dataframe[f'{TF}_ha_close'] < dataframe[f'{TF}_ha_open'].shift(1 * SHIFT)) &
            (dataframe[f'{TF}_ha_close'].shift(1 * SHIFT) > dataframe[f'{TF}_ha_open'].shift(2 * SHIFT))
        ).astype(int)

        dataframe.fillna(0, inplace=True)
        logger.info(f"{metadata['pair']} - enter_long signals: {dataframe['enter_long'].sum()}")
        logger.info(f"{metadata['pair']} entries: {dataframe['enter_long'].sum()}, exits: {dataframe['exit_long'].sum()}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[dataframe['exit_long'] == 1, 'exit_long'] = 1
        return dataframe
