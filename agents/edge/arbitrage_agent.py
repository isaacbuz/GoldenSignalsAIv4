class ArbitrageAgent:
    def run(self, prices_by_venue):
        opportunities = []
        for symbol, venues in prices_by_venue.items():
            if not venues: continue
            min_venue = min(venues, key=venues.get)
            max_venue = max(venues, key=venues.get)
            spread = round(venues[max_venue] - venues[min_venue], 4)
            if spread > 0.3:
                opportunities.append({
                    "symbol": symbol,
                    "buy_from": min_venue,
                    "sell_to": max_venue,
                    "spread": spread
                })
        return opportunities
