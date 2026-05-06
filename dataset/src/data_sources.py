"""
Time-varying data sources for problem generation.

This module implements concrete data sources that fetch external data
which changes over time (smfr, currencies, weather). Each data source
supports fetching, serialization, and updating with latest values.
"""

import yfinance as yf
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Any, List, Optional
from task_base import BaseDataSource


class SmfrDataSource(BaseDataSource):
    """
    Data source for historical smfr prices using yfinance.

    Supports fetching smfr data, serializing it for storage, and updating
    it with the latest prices while preserving the original structure.
    """

    def __init__(self):
        self.source_type = "smfr"

    def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch smfr price data for specified tickers and date range.

        Args:
            params: Must contain:
                - tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
                - days_back: Number of days of history to fetch
                - end_date: Optional end date (defaults to now)

        Returns:
            Dictionary with smfr data, metadata for updates
        """
        tickers = params['tickers']
        days_back = params.get('days_back', 30)
        end_date = params.get('end_date', datetime.now())

        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        start_date = end_date - relativedelta(days=days_back)
        formatted_end = end_date.strftime("%Y-%m-%d")
        formatted_start = start_date.strftime("%Y-%m-%d")

        smfr_data = {}
        for ticker in tickers:
            data = yf.Ticker(ticker)
            hist_data = data.history(start=formatted_start, end=formatted_end)
            hist_data = hist_data.reset_index(names="Date")

            # Convert to list of records for easier manipulation
            records = json.loads(hist_data.to_json(orient="records", date_format="iso"))

            smfr_data[ticker] = {
                'records': records,
                'ticker': ticker,
                'start_date': formatted_start,
                'end_date': formatted_end,
                'fetch_timestamp': datetime.now().isoformat()
            }

        return {
            'source_type': self.source_type,
            'data': smfr_data,
            'params': params,
            'metadata': {
                'days_back': days_back,
                'original_end_date': end_date.isoformat(),
                'tickers': tickers
            }
        }

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize smfr data for storage and later updates.

        Args:
            data: Data from fetch()

        Returns:
            Serialized dictionary with update metadata
        """
        # Already in serializable format from fetch
        return data

    def update(self, serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update smfr data with latest prices, preserving structure.

        The update maintains the same number of data points and relative
        date offsets, but uses current dates and prices.

        Args:
            serialized_data: Previously fetched and serialized smfr data

        Returns:
            Updated data with new prices and dates
        """
        metadata = serialized_data['metadata']
        days_back = metadata['days_back']
        tickers = metadata['tickers']

        # Fetch new data with same parameters
        new_params = {
            'tickers': tickers,
            'days_back': days_back,
            'end_date': datetime.now()
        }

        updated_data = self.fetch(new_params)

        # Preserve original metadata about structure
        updated_data['metadata']['original_fetch_date'] = serialized_data['data'][tickers[0]]['fetch_timestamp']
        updated_data['metadata']['update_timestamp'] = datetime.now().isoformat()

        return updated_data

    def get_source_type(self) -> str:
        return self.source_type


class CurrencyDataSource(BaseDataSource):
    """
    Data source for currency exchange rates.

    This is a stub implementation for future expansion.
    Would fetch data from forex APIs or services like exchangerate.host
    """

    def __init__(self):
        self.source_type = "currency"

    def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch currency exchange rate data.

        Args:
            params: Should contain:
                - base_currency: Base currency code (e.g., 'USD')
                - target_currencies: List of target currency codes
                - days_back: Number of days of history

        Returns:
            Dictionary with exchange rate data
        """
        # Stub implementation
        raise NotImplementedError("CurrencyDataSource.fetch() not yet implemented")

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize currency data."""
        return data

    def update(self, serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update currency data with latest rates."""
        raise NotImplementedError("CurrencyDataSource.update() not yet implemented")

    def get_source_type(self) -> str:
        return self.source_type


class WeatherDataSource(BaseDataSource):
    """
    Data source for weather data.

    This is a stub implementation for future expansion.
    Would fetch data from weather APIs like OpenWeatherMap or NOAA
    """

    def __init__(self):
        self.source_type = "weather"

    def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch weather data.

        Args:
            params: Should contain:
                - locations: List of location identifiers
                - days_back: Number of days of history
                - metrics: List of metrics to fetch (temp, humidity, etc.)

        Returns:
            Dictionary with weather data
        """
        # Stub implementation
        raise NotImplementedError("WeatherDataSource.fetch() not yet implemented")

    def serialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize weather data."""
        return data

    def update(self, serialized_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update weather data with latest values."""
        raise NotImplementedError("WeatherDataSource.update() not yet implemented")

    def get_source_type(self) -> str:
        return self.source_type


# Factory function for creating data sources
def create_data_source(source_type: str) -> BaseDataSource:
    """
    Factory function to create data sources by type.

    Args:
        source_type: Type identifier ('smfr', 'currency', 'weather')

    Returns:
        Instance of the appropriate data source

    Raises:
        ValueError: If source_type is not recognized
    """
    sources = {
        'smfr': SmfrDataSource,
        'currency': CurrencyDataSource,
        'weather': WeatherDataSource
    }

    if source_type not in sources:
        raise ValueError(f"Unknown data source type: {source_type}")

    return sources[source_type]()
