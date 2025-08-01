from yahoofinance import HistoricalPrices
from datetime import datetime, timedelta

try:
    print("Testing yahoofinance connectivity for AAPL...")

    # Get data for the last 5 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    # Create HistoricalPrices request
    req = HistoricalPrices(
        instrument='AAPL',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        frequency='1d'  # Daily data
    )

    # Get data as DataFrame
    df = req.to_dfs()
    print("\nHistorical Data:")
    print(df)

    if df.empty:
        print("No data returned. Possible connectivity or API issue.")
    else:
        print("Successfully fetched data for AAPL.")

except Exception as e:
    print(f"Error fetching data for AAPL: {e}")
