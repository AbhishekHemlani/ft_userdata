from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class Alpha1Renko(IStrategy):
    timeframe = '15m'
    can_short = False
    minimal_roi = {
        "0": 0.3
    }
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.10
    trailing_stop_positive_offset = 0.15
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True

    startup_candle_count: int = 400

    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs):
        """
        Dynamically increase trailing stoploss if Renko + Volume signal confirms upside
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        # Only proceed if Renko and volume conditions are met
        row = dataframe.iloc[-1]
        renko_up = row['renko_trend'] > 0
        volume_spike = row['volume'] > dataframe['volume'].rolling(48).mean().iloc[-1] * 2

        if renko_up and volume_spike:
            return -0.10  # Tighten SL to 10%
        return -0.15     # Default SL

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=56)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        dataframe['renko_trend'] = np.where(dataframe['close'] > dataframe['open'].shift(1), 1,
                                            np.where(dataframe['close'] < dataframe['open'].shift(1), -1, 0))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['plus_di'] > dataframe['minus_di']) &
                (dataframe['rsi'] < 75) &
                (dataframe['sar'] > dataframe['open']) &
                (dataframe['renko_trend'] > 0)
            ),
            'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['renko_trend'] < 0)
            ),
            'exit_long'] = 1
        return dataframe
