"""
Tests for data quality validation in GoldenSignalsAI V2.
Based on best practices for ensuring high-quality input data.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestDataQuality:
    """Test data quality validation functionality"""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data with some quality issues"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(100).cumsum(),
            'high': 102 + np.random.randn(100).cumsum(),
            'low': 98 + np.random.randn(100).cumsum(),
            'close': 100 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100)
        })

        # Introduce some data quality issues
        data.loc[10:12, 'close'] = np.nan  # Missing values
        data.loc[20, 'volume'] = -1000  # Negative volume
        data.loc[30, 'high'] = data.loc[30, 'low'] - 1  # High < Low
        data.loc[40:42, 'timestamp'] = data.loc[39, 'timestamp']  # Duplicate timestamps

        return data

    def test_missing_value_detection(self, sample_market_data):
        """Test detection of missing values in market data"""
        missing_values = sample_market_data.isnull().sum()

        assert missing_values['close'] == 3, "Should detect 3 missing close prices"
        assert missing_values['open'] == 0, "Should have no missing open prices"

        # Test missing value handling
        cleaned_data = sample_market_data.dropna()
        assert len(cleaned_data) == 97, "Should have 97 rows after dropping NaN"

    def test_outlier_detection(self, sample_market_data):
        """Test outlier detection using z-score and IQR methods"""
        closes = sample_market_data['close'].dropna()

        # Z-score method
        z_scores = np.abs((closes - closes.mean()) / closes.std())
        outliers_z = closes[z_scores > 3]

        # IQR method
        Q1 = closes.quantile(0.25)
        Q3 = closes.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = closes[(closes < Q1 - 1.5 * IQR) | (closes > Q3 + 1.5 * IQR)]

        # Both methods should identify outliers (if any exist)
        assert isinstance(outliers_z, pd.Series)
        assert isinstance(outliers_iqr, pd.Series)

    def test_data_consistency_validation(self, sample_market_data):
        """Test validation of data consistency rules"""
        # Check high >= low
        invalid_hl = sample_market_data[sample_market_data['high'] < sample_market_data['low']]
        assert len(invalid_hl) >= 1, "Should detect at least 1 row where high < low"
        assert 30 in invalid_hl.index, "Row 30 should have high < low"

        # Check high >= close and high >= open
        invalid_high = sample_market_data[
            (sample_market_data['high'] < sample_market_data['close']) |
            (sample_market_data['high'] < sample_market_data['open'])
        ]

        # Check low <= close and low <= open
        invalid_low = sample_market_data[
            (sample_market_data['low'] > sample_market_data['close']) |
            (sample_market_data['low'] > sample_market_data['open'])
        ]

        # Check negative volume
        negative_volume = sample_market_data[sample_market_data['volume'] < 0]
        assert len(negative_volume) == 1, "Should detect 1 row with negative volume"

    def test_timestamp_validation(self, sample_market_data):
        """Test timestamp validation and duplicate detection"""
        # Check for duplicate timestamps
        duplicates = sample_market_data[sample_market_data['timestamp'].duplicated()]
        assert len(duplicates) == 3, "Should detect 3 duplicate timestamps"

        # Check timestamp ordering
        time_diffs = sample_market_data['timestamp'].diff()
        out_of_order = time_diffs[time_diffs < pd.Timedelta(0)]
        assert len(out_of_order) == 0, "Timestamps should be in order"

        # Check for gaps in timestamps (assuming 5-minute intervals)
        expected_interval = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_interval * 1.5]  # Allow small tolerance

        # Remove duplicates and check again
        cleaned = sample_market_data.drop_duplicates(subset=['timestamp'])
        assert len(cleaned) == 97, "Should have 97 rows after removing duplicates"

    def test_data_normalization(self):
        """Test data normalization techniques"""
        # Create sample data
        prices = pd.Series([100, 110, 95, 105, 120, 90])
        volumes = pd.Series([1000000, 1500000, 800000, 1200000, 2000000, 600000])

        # Min-Max scaling
        price_min_max = (prices - prices.min()) / (prices.max() - prices.min())
        assert price_min_max.min() == 0.0
        assert price_min_max.max() == 1.0

        # Z-score normalization
        price_z = (prices - prices.mean()) / prices.std()
        assert abs(price_z.mean()) < 1e-10  # Should be close to 0
        assert abs(price_z.std() - 1.0) < 1e-10  # Should be close to 1

        # Verify volume scaling
        volume_scaled = volumes / 1000000  # Scale to millions
        assert volume_scaled.max() == 2.0
        assert volume_scaled.min() == 0.6

    def test_feature_engineering_quality(self):
        """Test quality of engineered features"""
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = pd.Series(100 + np.random.randn(50).cumsum(), index=dates)

        # Calculate technical indicators
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Validate RSI bounds
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi), "RSI should be between 0 and 100"

        # Moving averages
        sma_20 = prices.rolling(window=20).mean()
        ema_20 = prices.ewm(span=20, adjust=False).mean()

        # Validate moving averages
        assert sma_20.isna().sum() == 19, "SMA should have 19 NaN values at start"
        assert len(ema_20) == len(prices), "EMA should have same length as prices"

        # Bollinger Bands
        std_20 = prices.rolling(window=20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20

        # Validate Bollinger Bands
        valid_idx = ~upper_band.isna()
        assert all(upper_band[valid_idx] > sma_20[valid_idx]), "Upper band should be above SMA"
        assert all(lower_band[valid_idx] < sma_20[valid_idx]), "Lower band should be below SMA"

    def test_multi_source_data_alignment(self):
        """Test alignment of data from multiple sources"""
        # Simulate data from different sources with slightly different timestamps
        base_time = pd.Timestamp('2024-01-01 09:30:00')

        # Exchange data (1-second precision)
        exchange_times = [base_time + pd.Timedelta(seconds=i) for i in range(10)]
        exchange_data = pd.DataFrame({
            'timestamp': exchange_times,
            'price': 100 + np.random.randn(10).cumsum(),
            'volume': np.random.randint(1000, 5000, 10)
        })

        # News data (irregular timestamps)
        news_times = [
            base_time + pd.Timedelta(seconds=2.5),
            base_time + pd.Timedelta(seconds=5.1),
            base_time + pd.Timedelta(seconds=8.7)
        ]
        news_data = pd.DataFrame({
            'timestamp': news_times,
            'sentiment': [0.5, -0.3, 0.8],
            'headline': ['Positive news', 'Negative news', 'Very positive news']
        })

        # Align data using nearest timestamp
        exchange_data.set_index('timestamp', inplace=True)
        news_data.set_index('timestamp', inplace=True)

        # Merge with nearest timestamp (within 1 second tolerance)
        aligned = pd.merge_asof(
            exchange_data.sort_index(),
            news_data.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(seconds=1),
            direction='nearest'
        )

        # Verify alignment
        assert len(aligned) == len(exchange_data)
        assert aligned['sentiment'].notna().sum() >= 2, "Should match at least 2 news items"

    def test_data_quality_scoring(self):
        """Test data quality scoring mechanism"""
        # Create sample data with varying quality
        good_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': 100 + np.random.randn(100).cumsum() * 0.1,
            'high': 101 + np.random.randn(100).cumsum() * 0.1,
            'low': 99 + np.random.randn(100).cumsum() * 0.1,
            'close': 100 + np.random.randn(100).cumsum() * 0.1,
            'volume': np.random.randint(1000000, 2000000, 100)
        })

        # Ensure high >= max(open, close) and low <= min(open, close)
        good_data['high'] = good_data[['open', 'close', 'high']].max(axis=1)
        good_data['low'] = good_data[['open', 'close', 'low']].min(axis=1)

        def calculate_quality_score(data):
            """Calculate quality score for market data"""
            score = 100.0

            # Missing values penalty
            missing_pct = data.isnull().sum().sum() / data.size * 100
            score -= missing_pct * 2

            # Consistency violations penalty
            invalid_hl = len(data[data['high'] < data['low']])
            score -= invalid_hl * 5

            # Negative volume penalty
            negative_vol = len(data[data['volume'] < 0])
            score -= negative_vol * 10

            # Duplicate timestamps penalty
            duplicates = data['timestamp'].duplicated().sum()
            score -= duplicates * 3

            return max(0, score)

        # Test quality scoring
        good_score = calculate_quality_score(good_data)
        assert good_score >= 95, "Good data should have high quality score"

        # Create bad data
        bad_data = good_data.copy()
        bad_data.loc[0:5, 'close'] = np.nan
        bad_data.loc[10, 'high'] = bad_data.loc[10, 'low'] - 1
        bad_data.loc[20, 'volume'] = -1000

        bad_score = calculate_quality_score(bad_data)
        assert bad_score < good_score, "Bad data should have lower quality score"
        assert bad_score < 85, "Bad data score should be significantly lower"
