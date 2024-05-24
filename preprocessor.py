import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import inspect
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, weather_df_path='./Data/weather_per_fips.csv', counties_df_path='./Data/counties.csv',
                 default_steps=True):
        self.weather_df_path = weather_df_path
        self.counties_df = pd.read_csv(counties_df_path)
        self.steps = []
        self.available_steps = {name.replace("_step_", ""): method for name, method in inspect.getmembers(self)
                                if name.startswith('_step')}
        if default_steps:
            self.add_step('drop_columns', 'extract_date_info', 'extract_season', 'preprocess_discovery_time',
                          'get_county_fips', 'get_demographic_data', 'add_weather_data', 'transform_date_to_cyclic')

    def add_step(self, *args):
        """
        Adds a predefined step to the preprocessing pipeline by name.
        """
        for step_name in args:
            if step_name in self.available_steps:
                self.steps.append(self.available_steps[step_name])
            else:
                raise ValueError(f"No step named '{step_name}' found.")

    def transform(self, data):
        for step in self.steps:
            data = step(data)

        return data

    def fit(self, data, y=None):
        return self

    @staticmethod
    def define_features():
        numeric_features = ['LATITUDE', 'LONGITUDE', 'DISCOVERY_TIMESTAMP', 'discovery_hour_sin', 'discovery_hour_cos',
                            'DISCOVERY_MONTH_sin', 'DISCOVERY_MONTH_cos', 'DISCOVERY_YEAR', 'DISCOVERY_DAY_sin',
                            'DISCOVERY_DAY_cos', 'DISCOVERY_DOY_sin', 'DISCOVERY_DOY_cos', 'FIRE_SIZE', 'TEMPERATURE',
                            'PRECIPITATION', 'WIND']
        categorical_features = ['FIRE_SIZE_CLASS', 'SEASON', 'FIPS_CODE', 'OWNER_CODE', 'SOURCE_SYSTEM', 'SOURCE_SYSTEM_TYPE', 'STATE',
                                'IS_WEEKDAY']
        return numeric_features, categorical_features

    @staticmethod
    def _step_drop_columns(data):
        columns_to_drop = ['NWCG_REPORTING_AGENCY', 'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME',
                            'SOURCE_REPORTING_UNIT', 'SOURCE_REPORTING_UNIT_NAME', 'FIRE_CODE', 'COMPLEX_NAME',
                            'CONT_DATE', 'CONT_DOY', 'CONT_TIME', 'FIRE_NAME', 'FIRE_YEAR', 'MTBS_ID',
                            'LOCAL_INCIDENT_ID', 'LOCAL_FIRE_REPORT_ID', 'FOD_ID', 'FPA_ID', 'OBJECTID',
                            'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_FIRE_NAME', 'FIPS_NAME',
                            'OWNER_DESCR', 'Shape', 'COUNTY']
        data = data.drop(columns_to_drop, axis=1)
        return data

    @staticmethod
    def _step_extract_date_info(data):
        data['DISCOVERY_DATE'] = pd.to_datetime(data['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')
        data['DISCOVERY_YEAR'] = data['DISCOVERY_DATE'].dt.year
        data['DISCOVERY_MONTH'] = data['DISCOVERY_DATE'].dt.month
        data['DISCOVERY_DAY'] = data['DISCOVERY_DATE'].dt.day
        data['DISCOVERY_TIMESTAMP'] = data['DISCOVERY_DATE'].apply(lambda x: x.timestamp())
        data['IS_WEEKDAY'] = (data['DISCOVERY_DATE'].dt.weekday < 5).astype(int)  # 1 if weekday, 0 if weekend
        data.drop('DISCOVERY_DATE', axis=1, inplace=True)  # Drop the original 'DISCOVERY_DATE' column
        return data

    @staticmethod
    def _step_extract_season(data):
        def get_season(month):
            if 3 <= month <= 5:
                return 'Spring'
            elif 6 <= month <= 8:
                return 'Summer'
            elif 9 <= month <= 11:
                return 'Autumn'
            else:
                return 'Winter'
        data['SEASON'] = data['DISCOVERY_MONTH'].astype(int).apply(get_season)
        return data

    @staticmethod
    def _step_preprocess_discovery_time(data):
        # Replace NaN values with 0
        data['DISCOVERY_TIME'].fillna(0, inplace=True)

        # Now you can safely convert to integer
        data['DISCOVERY_TIME'] = data['DISCOVERY_TIME'].astype(int)
        data['DISCOVERY_TIME'] = data['DISCOVERY_TIME'].astype(str)

        # Pad with 0 only for values from length of 3
        data['DISCOVERY_TIME'] = data['DISCOVERY_TIME'].apply(lambda x: x.zfill(4) if len(x) == 3 else x)

        # find mode without 0
        mode = data['DISCOVERY_TIME'].replace('0', np.nan).mode()[0]

        # fill all values from length of 1 and 2 with mode
        data['DISCOVERY_TIME'] = data['DISCOVERY_TIME'].apply(
            lambda x: mode if not x or len(x) == 1 or len(x) == 2 else x)

        # Extract hour and minute from DISCOVERY_TIME
        data['discovery_hour'] = data['DISCOVERY_TIME'].str.slice(0, 2).astype(int)
        data['discovery_minute'] = data['DISCOVERY_TIME'].str.slice(2, 4).astype(int)

        # Check for incorrect time values
        incorrect_hours = data['discovery_hour'] > 23
        incorrect_minutes = data['discovery_minute'] > 59

        if incorrect_hours.any() or incorrect_minutes.any():
            # Handle incorrect time values by finding the mode of the hour and minute
            mode_hour = data.loc[~incorrect_hours, 'discovery_hour'].mode()[0]
            mode_minute = data.loc[~incorrect_minutes, 'discovery_minute'].mode()[0]
            # Replace incorrect values with the mode
            data.loc[incorrect_hours, 'discovery_hour'] = mode_hour
            data.loc[incorrect_minutes, 'discovery_minute'] = mode_minute
            # Print the numbers of incorrect hours and minutes after replacement
            print(f"incorrect_hours: {incorrect_hours.sum()}, incorrect_minutes: {incorrect_minutes.sum()}")

        # Convert hour and minute to cyclic features
        data['discovery_hour_sin'] = np.sin(2 * np.pi * data['discovery_hour'] / 24)
        data['discovery_hour_cos'] = np.cos(2 * np.pi * data['discovery_hour'] / 24)
        data['discovery_minute_sin'] = np.sin(2 * np.pi * data['discovery_minute'] / 60)
        data['discovery_minute_cos'] = np.cos(2 * np.pi * data['discovery_minute'] / 60)

        # Drop the original DISCOVERY_TIME feature
        data = data.drop(columns=['DISCOVERY_TIME', 'discovery_minute', 'discovery_hour', 'discovery_minute_sin',
                                  'discovery_minute_cos'])
        return data

    def _step_get_demographic_data(self, data):
        # Convert 'FIPS_CODE' to string in both DataFrames
        data['FIPS_CODE'] = data['FIPS_CODE'].astype(str)
        counties_df = self.counties_df.rename(columns={'fips': 'FIPS_CODE'})
        counties_df['FIPS_CODE'] = counties_df['FIPS_CODE'].astype(str)
        selected_cols = [
            'bls/2020/labor_force', 'bls/2020/employed', 'bls/2020/unemployed', 'race/non_hispanic_white_alone_male',
            'race/non_hispanic_white_alone_female',
            'race/black_alone_male', 'race/black_alone_female', 'race/asian_alone_male', 'race/asian_alone_female',
            'race/hispanic_male', 'race/hispanic_female',
            'age/0-4', 'age/5-9', 'age/10-14', 'age/15-19', 'age/20-24', 'age/25-29', 'age/30-34', 'age/35-39',
            'age/40-44', 'age/45-49', 'age/50-54', 'age/55-59', 'age/60-64', 'age/65-69', 'age/70-74', 'age/75-79',
            'age/80-84', 'age/85+',
            'male', 'female', 'population/2019'
        ]
        counties_df = counties_df[selected_cols + ['FIPS_CODE']]

        # Merge based on 'FIPS_CODE' column
        merged_df = pd.merge(data, counties_df, on='FIPS_CODE', how='inner')

        # Fill null values with the mean
        merged_df[selected_cols] = merged_df[selected_cols].fillna(merged_df[selected_cols].mean())
        return merged_df

    def _step_get_county_fips(self, data):
        # Extract the latitude and longitude columns for both datasets
        # Rename the latitude and longitude columns in the counties dataset to match the original dataset
        self.counties_df.rename(columns={'latitude (deg)': 'LATITUDE', 'longitude (deg)': 'LONGITUDE'}, inplace=True)

        original_locations = data[['LATITUDE', 'LONGITUDE']].dropna()
        counties_locations = self.counties_df[['LATITUDE', 'LONGITUDE']]

        # Initialize the NearestNeighbors model
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')

        # Fit the model to the counties locations
        nn.fit(counties_locations)

        # Find the nearest county for each point in the original dataset
        distances, indices = nn.kneighbors(original_locations)

        # Map the indices to FIPS codes
        nearest_fips = self.counties_df.iloc[indices.flatten()]['fips'].values

        # Since the indices returned by NearestNeighbors correspond to the rows in original_locations,
        # we need to ensure we're updating the correct rows in the original DataFrame.
        # Create a temporary DataFrame to hold the nearest FIPS codes with the correct indices
        temp_df = pd.DataFrame(nearest_fips, index=original_locations.index, columns=['nearest_fips'])

        # Fill in missing FIPS codes in the original dataset
        data.loc[data['FIPS_CODE'].isnull(), 'FIPS_CODE'] = temp_df['nearest_fips']

        # Convert FIPS_CODE to string
        data['FIPS_CODE'] = data['FIPS_CODE'].astype(int).astype(str)

        # Fill with leading 0
        data['FIPS_CODE'] = data['FIPS_CODE'].apply(lambda x: x.zfill(5))

        # create a dictionary to map state abbreviations to FIPS codes
        state_fips_map = {
            'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08', 'CT': '09', 'DE': '10',
            'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20',
            'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
            'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34', 'NM': '35', 'NY': '36',
            'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
            'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
            'WI': '55', 'WY': '56'
        }

        # Update rows with FIPS codes that start with '00' based on their state
        mask = data['FIPS_CODE'].str.startswith('00')
        data.loc[mask, 'FIPS_CODE'] = data.loc[mask, 'STATE'].map(state_fips_map) + data.loc[mask, 'FIPS_CODE'].str[2:]

        return data

    @staticmethod
    def _step_transform_date_to_cyclic(data):
        def transform_date_to_cyclic(df, column, max_val):
            df[column + '_sin'] = np.sin(2 * np.pi * df[column] / max_val)
            df[column + '_cos'] = np.cos(2 * np.pi * df[column] / max_val)
            df = df.drop(column, axis=1)
            return df

        data = transform_date_to_cyclic(data, 'DISCOVERY_DOY', 365)
        data = transform_date_to_cyclic(data, 'DISCOVERY_MONTH', 12)
        data = transform_date_to_cyclic(data, 'DISCOVERY_DAY', 31)
        return data

    def _step_add_weather_data(self, data):
        weather_df = pd.read_csv(self.weather_df_path)
        # Merge the weather data with the original DataFrame on the 'FIPS_CODE' and 'DISCOVERY_MONTH' columns
        # rename weather_df columns names to match the original df
        weather_df.rename(columns={'fips': 'FIPS_CODE', 'month': 'DISCOVERY_MONTH'}, inplace=True)

        # convert the 'FIPS_CODE' and 'DISCOVERY_MONTH' columns to string
        for col in ['FIPS_CODE', 'DISCOVERY_MONTH']:
            weather_df[col] = weather_df[col].astype(str)
            data[col] = data[col].astype(str)

        # merge the two dataframes
        data = pd.merge(data, weather_df, on=['FIPS_CODE', 'DISCOVERY_MONTH'], how='inner')

        # Convert the 'FIPS_CODE' and 'DISCOVERY_MONTH' columns to numeric
        for col in ['FIPS_CODE', 'DISCOVERY_MONTH']:
            weather_df[col] = weather_df[col].astype(float)
            data[col] = data[col].astype(float)

        # Rename the new columns
        data.rename(columns={'mean_temp': 'TEMPERATURE', 'precipitation': 'PRECIPITATION', 'wind_speed': 'WIND'},
                    inplace=True)
        return data
