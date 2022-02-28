RAW_DATA_DIR = '/Users/vcerqueira/Downloads/IWaveBNetwork_67d2_89f3_d3cb.csv'
DATA_DIR = 'wave_buoy_data.csv'

# Picked a buoy with a large number of data point and least missing values.
BUOY_ID = 'AMETS Berth B Wave Buoy'

UNUSED_COLUMNS = ['latitude', 'longitude', 'station_id',
                  'Hmax', 'THmax', 'MeanCurDirTo',
                  'MeanCurSpeed', 'SeaTemperature', 'PeakDirection']

EMBED_DIM = 5
HORIZON = 16
TARGET = 'SignificantWaveHeight'
THRESHOLD_PERCENTILE = 0.95
MC_N_TRIALS = 1000
CV_N_FOLDS = 10
