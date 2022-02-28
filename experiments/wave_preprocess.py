import pandas as pd

from wave_height.config import RAW_DATA_DIR, BUOY_ID, UNUSED_COLUMNS

wave = pd.read_csv(RAW_DATA_DIR)
# skipping 1st row
wave = wave[1:]

# filtering target buoy
buoy_wave = wave.loc[wave['station_id'] == BUOY_ID, :]

# casting time to appropriate format
buoy_wave.loc[:, 'time'] = pd.to_datetime(buoy_wave['time'])
# setting time as index
buoy_wave.set_index('time', inplace=True)
buoy_wave = buoy_wave.sort_index()
# removing useless/unused columns
buoy_wave = buoy_wave.loc[:, ~buoy_wave.columns.str.endswith('_qc')]
buoy_wave = buoy_wave.drop(UNUSED_COLUMNS, axis=1)
# casting columns as floats
buoy_wave = buoy_wave.astype(float)

buoy_wave.to_csv('wave_buoy_data.csv')
# sample to github
buoy_wave.head(500).to_csv('wave_buoy_data_sample.csv')
