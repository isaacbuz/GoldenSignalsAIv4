# Timezone Issues Report

## Summary
- Total files with issues: 141
- Total issues found: 426

## Files with Issues

### agents/arbitrage/base.py
- Line 31: datetime.now() - should use now_utc() - self.timestamp = timestamp or datetime.now().timestamp()

### agents/arbitrage/cross_exchange.py
- Line 102: datetime.now() - should use now_utc() - timestamp=datetime.now().timestamp()

### agents/arbitrage/execution.py
- Line 175: datetime.now() - should use now_utc() - "timestamp": datetime.now().timestamp(),
- Line 257: datetime.now() - should use now_utc() - "timestamp": datetime.now().timestamp(),

### agents/arbitrage/statistical.py
- Line 121: datetime.now() - should use now_utc() - timestamp=datetime.now().timestamp()
- Line 130: datetime.now() - should use now_utc() - timestamp=datetime.now().timestamp()

### agents/base.py
- Line 177: datetime.utcnow() - should use now_utc() - self.performance.last_updated = datetime.utcnow()
- Line 305: datetime.utcnow() - should use now_utc() - self._state["last_analysis_time"] = datetime.utcnow().isoformat()
- Line 337: datetime.utcnow() - should use now_utc() - since = datetime.utcnow() - timedelta(hours=limit)
- Line 373: datetime.utcnow() - should use now_utc() - since = datetime.utcnow() - timedelta(hours=24)  # Last 24 hours

### agents/common/base/enhanced_base_agent.py
- Line 373: datetime.now() - should use now_utc() - self.metrics.last_execution = datetime.now()
- Line 474: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 520: datetime.now() - should use now_utc() - test_data = {"test": True, "timestamp": datetime.now().isoformat()}

### agents/common/data_bus.py
- Line 35: datetime.now() - should use now_utc() - timestamp = datetime.now()
- Line 77: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 118: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 149: datetime.now() - should use now_utc() - 'age_seconds': (datetime.now() - entry['timestamp']).total_seconds()
- Line 156: datetime.now() - should use now_utc() - age = (datetime.now() - entry['timestamp']).total_seconds()

### agents/common/hybrid_agent_base.py
- Line 210: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 298: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 316: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 333: datetime.now() - should use now_utc() - age = (datetime.now() - data['timestamp']).total_seconds()

### agents/common/templates/base_agent_template.py
- Line 109: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),

### agents/common/utils/performance_monitor.py
- Line 77: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### agents/common/utils/signal_logger.py
- Line 34: datetime.now() - should use now_utc() - timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
- Line 44: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/flow/order_flow_agent.py
- Line 269: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/market/market_profile_agent.py
- Line 268: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/options/simple_options_flow_agent.py
- Line 316: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/risk/real_time_monitor.py
- Line 155: pd.Timestamp.now() - should use now_utc() - 'timestamp': pd.Timestamp.now(),

### agents/core/sentiment/news_agent.py
- Line 148: datetime.now() - should use now_utc() - current_time = datetime.now()
- Line 258: datetime.now() - should use now_utc() - hours_ago = (datetime.now() - news_time).total_seconds() / 3600

### agents/core/sentiment/simple_sentiment_agent.py
- Line 270: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/adx_agent.py
- Line 211: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/atr_agent.py
- Line 175: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/bollinger_bands_agent.py
- Line 128: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/ema_agent.py
- Line 158: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/enhanced_pattern_agent.py
- Line 489: datetime.now() - should use now_utc() - hour = datetime.now().hour
- Line 491: datetime.now() - should use now_utc() - features.append(float(datetime.now().month) / 12.0)  # Normalized month

### agents/core/technical/enhanced_volume_spike_agent.py
- Line 177: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 287: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 327: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### agents/core/technical/fibonacci_agent.py
- Line 198: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/ichimoku_agent.py
- Line 178: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/ma_crossover_agent.py
- Line 82: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/macd_agent.py
- Line 88: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/parabolic_sar_agent.py
- Line 240: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/simple_working_agent.py
- Line 153: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/std_dev_agent.py
- Line 193: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/stochastic_agent.py
- Line 151: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/volume_spike_agent.py
- Line 76: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/technical/vwap_agent.py
- Line 44: datetime.now() - should use now_utc() - now = datetime.now().time()
- Line 169: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/core/volume/volume_profile_agent.py
- Line 279: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/edge/notify_agent.py
- Line 35: datetime.utcnow() - should use now_utc() - "timestamp": signal.get("timestamp") or datetime.utcnow().isoformat()

### agents/hybrid/hybrid_sentiment_flow_agent.py
- Line 313: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 323: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 332: datetime.now() - should use now_utc() - 'timestamp': datetime.now()

### agents/infrastructure/monitoring/monitoring_agents.py
- Line 56: datetime.now() - should use now_utc() - current_time = datetime.now()

### agents/meta/ml_meta_agent.py
- Line 41: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 331: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/meta/simple_consensus_agent.py
- Line 124: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/ml/transformer_agent.py
- Line 252: datetime.utcnow() - should use now_utc() - 'timestamp': datetime.utcnow(),

### agents/ml_agents/base_ml_agent.py
- Line 182: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 202: datetime.now() - should use now_utc() - self.last_training_time = datetime.now()
- Line 241: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 243: datetime.now() - should use now_utc() - timestamp=datetime.now()
- Line 263: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 309: datetime.now() - should use now_utc() - time_since_training = (datetime.now() - self.last_training_time).days

### agents/ml_agents/lstm_agent.py
- Line 161: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### agents/optimization/performance_tracker.py
- Line 44: pd.Timestamp.now() - should use now_utc() - "timestamp": pd.Timestamp.now(),

### agents/orchestration/hybrid_orchestrator.py
- Line 163: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 250: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 256: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 300: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()

### agents/orchestration/simple_orchestrator.py
- Line 179: datetime.now() - should use now_utc() - consensus_signal['timestamp'] = datetime.now().isoformat()
- Line 226: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 282: datetime.now() - should use now_utc() - "timestamp": signal.get('timestamp', datetime.now().isoformat())

### agents/orchestrator.py
- Line 186: datetime.utcnow() - should use now_utc() - "last_updated": datetime.utcnow()
- Line 236: datetime.utcnow() - should use now_utc() - current_time = datetime.utcnow()
- Line 292: datetime.utcnow() - should use now_utc() - self.last_signals[symbol] = datetime.utcnow()
- Line 398: datetime.utcnow() - should use now_utc() - expires_at=datetime.utcnow() + timedelta(hours=2)
- Line 583: datetime.utcnow() - should use now_utc() - self.agent_performance[agent_name]["last_updated"] = datetime.utcnow()
- Line 611: datetime.utcnow() - should use now_utc() - "last_updated": datetime.utcnow().isoformat()

### agents/quant/signal_prophet_agent.py
- Line 382: datetime.now() - should use now_utc() - signal_time=datetime.now(),
- Line 383: datetime.now() - should use now_utc() - expiry_time=datetime.now() + timedelta(hours=4),
- Line 487: datetime.now() - should use now_utc() - current_hour = datetime.now().hour

### agents/research/ml/classifiers/ensemble_classifier_agent.py
- Line 139: pd.Timestamp.now() - should use now_utc() - "timestamp": pd.Timestamp.now().isoformat()

### agents/signals/arbitrage_signals.py
- Line 102: datetime.now() - should use now_utc() - signal_id=f"SPATIAL_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 103: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 259: datetime.now() - should use now_utc() - signal_id=f"STAT_PAIR_{asset1}_{asset2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 260: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 337: datetime.now() - should use now_utc() - signal_id=f"STAT_MEAN_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 338: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 408: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=self.lookback_period, freq='D')
- Line 452: datetime.now() - should use now_utc() - days_to_event = (event_data['date'] - datetime.now()).days
- Line 475: datetime.now() - should use now_utc() - days_to_event = (event_data['date'] - datetime.now()).days
- Line 485: datetime.now() - should use now_utc() - signal_id=f"RISK_EVENT_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 486: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 540: datetime.now() - should use now_utc() - signal_id=f"RISK_VOL_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 541: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### agents/signals/integrated_signal_system.py
- Line 52: datetime.now() - should use now_utc() - print(f"Scanning {len(symbols)} symbols at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
- Line 250: datetime.now() - should use now_utc() - 'generated_at': datetime.now().isoformat(),
- Line 319: datetime.now() - should use now_utc() - 'executed_at': datetime.now().isoformat(),
- Line 331: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()

### agents/signals/precise_options_signals.py
- Line 339: datetime.now() - should use now_utc() - signal_id=f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
- Line 340: datetime.now() - should use now_utc() - generated_at=datetime.now(),
- Line 436: datetime.now() - should use now_utc() - target_date = datetime.now() + timedelta(days=hold_days + 3)
- Line 448: datetime.now() - should use now_utc() - now = datetime.now()

### agents/signals/signal_manager.py
- Line 22: datetime.now() - should use now_utc() - self.timestamp = timestamp or datetime.now()
- Line 54: datetime.now() - should use now_utc() - cutoff = datetime.now().timestamp() - (minutes * 60)
- Line 63: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 81: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### agents/technical/momentum_agent.py
- Line 100: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 311: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 337: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 357: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 373: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 418: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 435: datetime.now() - should use now_utc() - 'timestamp': datetime.now()

### src/api/performance_dashboard.py
- Line 35: datetime.now() - should use now_utc() - "last_update": datetime.now().isoformat()
- Line 276: datetime.now() - should use now_utc() - "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),

### src/api/signal_api.py
- Line 60: datetime.now() - should use now_utc() - return {"status": "healthy", "timestamp": datetime.now().isoformat()}

### src/api/v1/ai_analyst.py
- Line 48: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 206: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 250: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 280: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 287: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 332: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()
- Line 438: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/api/v1/analytics.py
- Line 212: datetime.now() - should use now_utc() - report_date = date or datetime.now().strftime("%Y-%m-%d")

### src/api/v1/auth.py
- Line 308: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow().isoformat(),
- Line 309: datetime.utcnow() - should use now_utc() - last_login=datetime.utcnow().isoformat()

### src/api/v1/backtesting.py
- Line 164: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow()
- Line 216: datetime.utcnow() - should use now_utc() - start_date=datetime.utcnow() - timedelta(days=365),
- Line 217: datetime.utcnow() - should use now_utc() - end_date=datetime.utcnow(),
- Line 219: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow() - timedelta(hours=2)
- Line 266: datetime.utcnow() - should use now_utc() - start_date=datetime.utcnow() - timedelta(days=180),
- Line 267: datetime.utcnow() - should use now_utc() - end_date=datetime.utcnow(),
- Line 269: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow() - timedelta(days=1)
- Line 275: datetime.utcnow() - should use now_utc() - {"date": (datetime.utcnow() - timedelta(days=i)).isoformat(),
- Line 291: datetime.utcnow() - should use now_utc() - "entry_time": (datetime.utcnow() - timedelta(days=180-i)).isoformat(),
- Line 292: datetime.utcnow() - should use now_utc() - "exit_time": (datetime.utcnow() - timedelta(days=179-i)).isoformat()

### src/api/v1/hybrid_signals.py
- Line 42: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 108: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 191: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 224: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/api/v1/integrated_signals.py
- Line 129: datetime.now() - should use now_utc() - scan_timestamp=datetime.now().isoformat(),
- Line 150: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 230: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 328: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### src/api/v1/market_data.py
- Line 71: datetime.utcnow() - should use now_utc() - "timestamp": int(datetime.utcnow().timestamp() * 1000)  # milliseconds
- Line 147: datetime.utcnow() - should use now_utc() - "timestamp": int(datetime.utcnow().timestamp() * 1000)

### src/api/v1/notifications.py
- Line 130: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow() - timedelta(minutes=5),
- Line 132: datetime.utcnow() - should use now_utc() - delivered_at=datetime.utcnow() - timedelta(minutes=4),
- Line 154: datetime.utcnow() - should use now_utc() - notif_id = f"notif_{datetime.utcnow().timestamp()}"
- Line 175: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow(),
- Line 257: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow() - timedelta(days=7),
- Line 258: datetime.utcnow() - should use now_utc() - last_triggered=datetime.utcnow() - timedelta(hours=2),
- Line 276: datetime.utcnow() - should use now_utc() - alert_id = f"alert_{datetime.utcnow().timestamp()}"
- Line 289: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow(),

### src/api/v1/portfolio.py
- Line 190: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 201: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### src/api/v1/signals.py
- Line 85: datetime.utcnow() - should use now_utc() - since = datetime.utcnow() - timedelta(hours=hours_back)
- Line 230: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 317: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat()
- Line 349: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat()

### src/api/v1/websocket.py
- Line 82: datetime.utcnow() - should use now_utc() - "server_time": datetime.utcnow().isoformat()
- Line 84: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 144: datetime.utcnow() - should use now_utc() - "data": {"server_time": datetime.utcnow().isoformat()},
- Line 145: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 216: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 245: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 282: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 309: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 322: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 348: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 357: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()

### src/application/events/event_definitions.py
- Line 10: datetime.now() - should use now_utc() - timestamp: datetime = datetime.now()
- Line 18: datetime.now() - should use now_utc() - timestamp: datetime = datetime.now()

### src/application/events/price_alert_event.py
- Line 10: datetime.now() - should use now_utc() - timestamp: datetime = datetime.now()

### src/application/events/signal_event.py
- Line 10: datetime.now() - should use now_utc() - timestamp: datetime = datetime.now()

### src/core/auth.py
- Line 92: datetime.utcnow() - should use now_utc() - expire = datetime.utcnow() + expires_delta
- Line 94: datetime.utcnow() - should use now_utc() - expire = datetime.utcnow() + timedelta(minutes=settings.security.access_token_expire_minutes)
- Line 110: datetime.utcnow() - should use now_utc() - expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last 7 days
- Line 155: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow(),
- Line 156: datetime.utcnow() - should use now_utc() - last_login=datetime.utcnow()
- Line 172: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow()
- Line 189: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow()
- Line 228: datetime.utcnow() - should use now_utc() - created_at=datetime.utcnow()

### src/core/dependencies.py
- Line 121: datetime.utcnow() - should use now_utc() - if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
- Line 300: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),

### src/core/redis_manager.py
- Line 298: datetime.utcnow() - should use now_utc() - current_time = int(datetime.utcnow().timestamp())

### src/data/fetchers/live_data_fetcher.py
- Line 116: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 363: datetime.now() - should use now_utc() - if int(datetime.now().timestamp()) % 60 == 0:  # Every minute

### src/data/fetchers/realtime_fetcher.py
- Line 6: pd.Timestamp.now() - should use now_utc() - "timestamp": [pd.Timestamp.now()]

### src/domain/analytics/performance_tracker.py
- Line 24: datetime.now() - should use now_utc() - self._last_update = datetime.now()
- Line 35: datetime.now() - should use now_utc() - "timestamp": datetime.now(),
- Line 40: datetime.now() - should use now_utc() - if datetime.now() - self._last_update >= self._update_interval:
- Line 98: datetime.now() - should use now_utc() - self._last_update = datetime.now()

### src/domain/backtesting/advanced_backtest_engine.py
- Line 1063: datetime.now() - should use now_utc() - return pd.Series([self.config['initial_capital']], index=[datetime.now()])

### src/domain/backtesting/backtest_reporting.py
- Line 51: datetime.now() - should use now_utc() - timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
- Line 68: datetime.now() - should use now_utc() - "generated_at": datetime.now().isoformat(),

### src/domain/portfolio/portfolio_manager.py
- Line 63: datetime.now() - should use now_utc() - entry_date=datetime.now() if quantity > 0 else current_position.entry_date
- Line 70: datetime.now() - should use now_utc() - entry_date=datetime.now()
- Line 78: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),

### src/domain/signal_engine.py
- Line 134: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat(),
- Line 149: datetime.now() - should use now_utc() - self._last_signal_time[symbol] = datetime.now()
- Line 221: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat(),
- Line 559: datetime.now() - should use now_utc() - return datetime.now() - self._last_signal_time[symbol] < self._cache_duration
- Line 624: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 686: datetime.now() - should use now_utc() - "timestamp": datetime.now(),
- Line 1279: datetime.now() - should use now_utc() - next_earnings = earnings_data[earnings_data["date"] > datetime.now()].iloc[0]

### src/domain/trading/entities/trade.py
- Line 9: datetime.now() - should use now_utc() - timestamp: datetime = datetime.now()

### src/domain/trading/options_analysis.py
- Line 37: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 163: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/main.py
- Line 450: datetime.now() - should use now_utc() - "timestamp": int(datetime.now().timestamp())
- Line 589: datetime.now() - should use now_utc() - "id": signal.get("id", f"SIG{datetime.now().timestamp()}"),
- Line 650: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/main_simple.py
- Line 139: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 154: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 184: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 316: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 348: datetime.now() - should use now_utc() - now = datetime.now()
- Line 374: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 413: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 431: datetime.now() - should use now_utc() - now = datetime.now()
- Line 451: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 473: datetime.now() - should use now_utc() - "last_updated": datetime.now().isoformat(),
- Line 481: datetime.now() - should use now_utc() - "last_updated": datetime.now().isoformat(),
- Line 489: datetime.now() - should use now_utc() - "last_updated": datetime.now().isoformat(),
- Line 501: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 570: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/main_v2.py
- Line 100: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 117: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 158: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 246: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 264: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 326: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### src/ml/models/market_data.py
- Line 260: datetime.utcnow() - should use now_utc() - age = (datetime.utcnow() - self.last_updated).total_seconds() / 60

### src/ml/models/portfolio.py
- Line 150: datetime.utcnow() - should use now_utc() - self.updated_at = datetime.utcnow()
- Line 204: datetime.utcnow() - should use now_utc() - self.executed_at = datetime.utcnow()
- Line 210: datetime.utcnow() - should use now_utc() - self.updated_at = datetime.utcnow()

### src/ml/models/risk.py
- Line 154: datetime.utcnow() - should use now_utc() - self.acknowledged_at = datetime.utcnow()
- Line 159: datetime.utcnow() - should use now_utc() - self.resolved_at = datetime.utcnow()

### src/ml/models/signals.py
- Line 99: datetime.utcnow() - should use now_utc() - return datetime.utcnow() + timedelta(minutes=values["time_horizon"])
- Line 101: datetime.utcnow() - should use now_utc() - return datetime.utcnow() + timedelta(hours=24)  # Default 24 hour expiry
- Line 106: datetime.utcnow() - should use now_utc() - return self.expires_at and datetime.utcnow() > self.expires_at

### src/ml/models/users.py
- Line 169: datetime.utcnow() - should use now_utc() - return datetime.utcnow() > self.expires_at
- Line 201: datetime.utcnow() - should use now_utc() - return datetime.utcnow() > self.expires_at

### src/ml/research/classifiers/ensemble_classifier_agent.py
- Line 139: pd.Timestamp.now() - should use now_utc() - "timestamp": pd.Timestamp.now().isoformat()

### src/services/ai_chat_service.py
- Line 235: datetime.utcnow() - should use now_utc() - session.updated_at = datetime.utcnow()
- Line 578: datetime.utcnow() - should use now_utc() - "duration": (datetime.utcnow() - session.created_at).total_seconds(),

### src/services/ai_chat_service_enhanced.py
- Line 1008: datetime.utcnow() - should use now_utc() - 'timestamp': datetime.utcnow().isoformat()

### src/services/ai_trading_analyst.py
- Line 190: datetime.now() - should use now_utc() - *Analysis generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with {self._describe_confidence()} confidence*

### src/services/audit_logger.py
- Line 12: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),

### src/services/chart_vision_analyzer.py
- Line 594: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat()

### src/services/data_quality_validator.py
- Line 142: datetime.now() - should use now_utc() - end_date = datetime.now()
- Line 267: pd.Timestamp.now() - should use now_utc() - current_date = pd.Timestamp.now()

### src/services/data_service.py
- Line 24: pd.Timestamp.now() - should use now_utc() - "timestamp": pd.Timestamp.now().isoformat()

### src/services/decision_logger.py
- Line 11: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),

### src/services/live_data_service.py
- Line 106: datetime.utcnow() - should use now_utc() - self.stats["uptime_start"] = datetime.utcnow()
- Line 174: datetime.utcnow() - should use now_utc() - self.stats["last_update"] = datetime.utcnow()
- Line 224: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 243: datetime.utcnow() - should use now_utc() - uptime = datetime.utcnow() - self.stats["uptime_start"]
- Line 256: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 269: datetime.utcnow() - should use now_utc() - cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.cache_ttl)
- Line 330: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 349: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),
- Line 394: datetime.utcnow() - should use now_utc() - if cached and (datetime.utcnow() - datetime.fromisoformat(cached.get("timestamp", "")) < timedelta(seconds=10)):
- Line 414: datetime.utcnow() - should use now_utc() - "uptime": str(datetime.utcnow() - self.stats["uptime_start"]) if self.stats["uptime_start"] else "0:00:00",

### src/services/market_data_manager.py
- Line 28: datetime.now() - should use now_utc() - now = datetime.now()
- Line 53: datetime.now() - should use now_utc() - if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
- Line 66: datetime.now() - should use now_utc() - self.last_failure_time = datetime.now()
- Line 120: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 137: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 201: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 240: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 255: pd.Timestamp.now() - should use now_utc() - dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='D')

### src/services/market_data_service.py
- Line 287: datetime.now() - should use now_utc() - "timestamp": datetime.now().timestamp()
- Line 304: datetime.now() - should use now_utc() - age = datetime.now().timestamp() - cached["timestamp"]
- Line 314: datetime.now() - should use now_utc() - age = datetime.now().timestamp() - mtime
- Line 330: datetime.now() - should use now_utc() - "timestamp": datetime.now().timestamp()
- Line 342: datetime.now() - should use now_utc() - age = datetime.now().timestamp() - mtime
- Line 593: datetime.now() - should use now_utc() - now = datetime.now()
- Line 699: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()

### src/services/market_data_service_mock.py
- Line 104: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat()
- Line 122: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat()
- Line 140: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat(),
- Line 167: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat()
- Line 175: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
- Line 267: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat(),
- Line 320: datetime.now() - should use now_utc() - timestamp=datetime.now().isoformat(),
- Line 327: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),

### src/services/monitoring_service.py
- Line 318: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),

### src/services/multi_source_aggregator.py
- Line 70: datetime.now() - should use now_utc() - elapsed = (datetime.now() - self.last_request_time).total_seconds()
- Line 74: datetime.now() - should use now_utc() - self.last_request_time = datetime.now()
- Line 324: datetime.now() - should use now_utc() - cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
- Line 375: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### src/services/multi_timeframe_engine.py
- Line 23: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()

### src/services/performance_tracker.py
- Line 14: pd.Timestamp.utcnow() - should use now_utc() - "timestamp": pd.Timestamp.utcnow().isoformat()

### src/services/prediction_visualization.py
- Line 142: datetime.now() - should use now_utc() - 'prediction_timestamp': datetime.now().isoformat()
- Line 558: datetime.now() - should use now_utc() - current_time = datetime.now()

### src/services/rate_limit_handler.py
- Line 371: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),
- Line 407: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),

### src/services/signal_filtering_pipeline.py
- Line 177: datetime.now() - should use now_utc() - cutoff_time = datetime.now() - self.time_window

### src/services/signal_monitoring_service.py
- Line 242: datetime.now() - should use now_utc() - entry_time=datetime.now(),
- Line 273: datetime.now() - should use now_utc() - signal_outcome.exit_time = datetime.now()
- Line 312: datetime.now() - should use now_utc() - """, (signal_id, feedback_type, json.dumps(feedback_value), datetime.now().isoformat()))
- Line 326: datetime.now() - should use now_utc() - (datetime.now() - self.cache_timestamp).seconds < self.cache_ttl and
- Line 334: datetime.now() - should use now_utc() - cutoff_time = datetime.now() - timeframe
- Line 437: datetime.now() - should use now_utc() - self.cache_timestamp = datetime.now()
- Line 571: datetime.now() - should use now_utc() - """, (datetime.now().isoformat(), json.dumps(metrics.to_dict())))

### src/services/signal_service.py
- Line 276: datetime.utcnow() - should use now_utc() - "execution_time": datetime.utcnow(),
- Line 349: datetime.utcnow() - should use now_utc() - if s.created_at > datetime.utcnow() - timedelta(hours=24)

### src/utils/error_recovery.py
- Line 106: datetime.now() - should use now_utc() - datetime.now() - self.last_failure_time >= self.config.recovery_timeout
- Line 124: datetime.now() - should use now_utc() - self.last_failure_time = datetime.now()
- Line 235: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 299: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### src/utils/metrics.py
- Line 54: datetime.utcnow() - should use now_utc() - timestamp = datetime.utcnow()
- Line 141: datetime.utcnow() - should use now_utc() - cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
- Line 208: datetime.utcnow() - should use now_utc() - cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

### src/websocket/enhanced_websocket_service.py
- Line 73: datetime.utcnow() - should use now_utc() - 'uptime_start': datetime.utcnow()
- Line 135: datetime.utcnow() - should use now_utc() - last_heartbeat=datetime.utcnow(),
- Line 219: datetime.utcnow() - should use now_utc() - self.connection_status[connection_id].last_heartbeat = datetime.utcnow()
- Line 231: datetime.utcnow() - should use now_utc() - timestamp=datetime.utcnow(),
- Line 288: datetime.utcnow() - should use now_utc() - current_time = datetime.utcnow()
- Line 356: datetime.utcnow() - should use now_utc() - uptime = datetime.utcnow() - self.metrics['uptime_start']

### src/websocket/manager.py
- Line 35: datetime.utcnow() - should use now_utc() - self.connected_at = datetime.utcnow()
- Line 36: datetime.utcnow() - should use now_utc() - self.last_activity = datetime.utcnow()
- Line 108: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat(),
- Line 184: datetime.utcnow() - should use now_utc() - connection_info.last_activity = datetime.utcnow()
- Line 314: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 361: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 402: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 411: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 419: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 427: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 434: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 462: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 471: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 480: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 486: datetime.utcnow() - should use now_utc() - "timestamp": datetime.utcnow().isoformat()
- Line 535: datetime.utcnow() - should use now_utc() - current_time = datetime.utcnow()

### src/websocket/professional_websocket_service.py
- Line 112: datetime.now() - should use now_utc() - if datetime.now() - timestamp < self.ttl:
- Line 118: datetime.now() - should use now_utc() - self.cache[symbol] = (data, datetime.now())
- Line 122: datetime.now() - should use now_utc() - now = datetime.now()
- Line 212: datetime.now() - should use now_utc() - quote.get('t', datetime.now().isoformat())
- Line 309: datetime.now() - should use now_utc() - timestamp=datetime.now(),

### src/working_server.py
- Line 87: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 111: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 126: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 165: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()
- Line 180: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 213: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### tests/AlphaPy/test_stress.py
- Line 39: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 44: datetime.now() - should use now_utc() - processing_time = (datetime.now() - start_time).total_seconds()
- Line 64: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 66: datetime.now() - should use now_utc() - optimization_time = (datetime.now() - start_time).total_seconds()
- Line 93: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 111: datetime.now() - should use now_utc() - calculation_time = (datetime.now() - start_time).total_seconds()

### tests/agents/common/test_signal_logger.py
- Line 23: datetime.now() - should use now_utc() - base_time = datetime.now()
- Line 83: datetime.now() - should use now_utc() - base_time = datetime.now()

### tests/agents/test_backtest_engine.py
- Line 46: datetime.now() - should use now_utc() - timestamp=datetime.now()
- Line 57: datetime.now() - should use now_utc() - timestamp=datetime.now()
- Line 77: datetime.now() - should use now_utc() - timestamp=datetime.now()
- Line 117: datetime.now() - should use now_utc() - timestamp=datetime.now()

### tests/agents/test_orchestrator.py
- Line 132: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat()

### tests/conftest.py
- Line 107: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')

### tests/integration/test_signal_pipeline_integration.py
- Line 136: pd.Timestamp.now() - should use now_utc() - dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
- Line 207: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 209: datetime.now() - should use now_utc() - generation_time = (datetime.now() - start_time).total_seconds()
- Line 216: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 218: datetime.now() - should use now_utc() - filter_time = (datetime.now() - start_time).total_seconds()

### tests/production_data_test_framework.py
- Line 227: datetime.now() - should use now_utc() - end_date = datetime.now()
- Line 395: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 418: datetime.now() - should use now_utc() - results['latency_ms'] = (datetime.now() - start_time).total_seconds() * 1000
- Line 482: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 487: datetime.now() - should use now_utc() - latency_ms = (datetime.now() - start_time).total_seconds() * 1000
- Line 604: datetime.now() - should use now_utc() - f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

### tests/root_tests/test_all_agents.py
- Line 331: datetime.now() - should use now_utc() - print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
- Line 360: datetime.now() - should use now_utc() - print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

### tests/root_tests/test_hybrid_system.py
- Line 153: datetime.now() - should use now_utc() - 'timestamp': datetime.now().isoformat(),
- Line 322: datetime.now() - should use now_utc() - print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
- Line 339: datetime.now() - should use now_utc() - print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

### tests/root_tests/test_live_data.py
- Line 76: datetime.now() - should use now_utc() - print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

### tests/root_tests/test_live_data_and_backtest.py
- Line 264: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 274: datetime.now() - should use now_utc() - elapsed = (datetime.now() - start_time).total_seconds()

### tests/root_tests/test_live_data_simple.py
- Line 28: datetime.now() - should use now_utc() - 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
- Line 109: datetime.now() - should use now_utc() - print(f"Update Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

### tests/root_tests/test_signal_generation.py
- Line 322: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),

### tests/root_tests/test_system.py
- Line 120: datetime.now() - should use now_utc() - start_time = datetime.now()
- Line 128: datetime.now() - should use now_utc() - end_time = datetime.now()

### tests/root_tests/test_yfinance_connectivity.py
- Line 8: datetime.now() - should use now_utc() - end_date = datetime.now()

### tests/test_comprehensive_system.py
- Line 54: datetime.now() - should use now_utc() - self.timestamp = datetime.now()
- Line 696: datetime.now() - should use now_utc() - "timestamp": datetime.now().isoformat(),
- Line 716: datetime.now() - should use now_utc() - report_path = f"logs/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

### tests/test_integration.py
- Line 25: datetime.now() - should use now_utc() - end_date = datetime.now()
- Line 35: datetime.now() - should use now_utc() - end_date = datetime.now()

### tests/unit/test_domain_risk_management.py
- Line 434: datetime.now() - should use now_utc() - current_hour = datetime.now().hour + datetime.now().minute / 60

### tests/unit/test_market_data_service.py
- Line 25: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
- Line 129: datetime.now() - should use now_utc() - times = pd.date_range(end=datetime.now(), periods=78, freq='5min')
- Line 221: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=10, freq='D')

### tests/unit/test_monitoring_feedback.py
- Line 22: datetime.now() - should use now_utc() - base_time = datetime.now()
- Line 43: datetime.now() - should use now_utc() - dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
- Line 107: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 115: datetime.now() - should use now_utc() - 'timestamp': datetime.now()
- Line 217: datetime.now() - should use now_utc() - self.last_retrain_date = datetime.now() - timedelta(days=30)
- Line 253: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 257: datetime.now() - should use now_utc() - self.last_retrain_date = datetime.now()
- Line 266: datetime.now() - should use now_utc() - datetime.now()
- Line 277: datetime.now() - should use now_utc() - datetime.now()
- Line 404: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),
- Line 556: datetime.now() - should use now_utc() - 'timestamp': datetime.now(),

### tests/unit/test_signal_filtering_pipeline.py
- Line 34: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 121: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 139: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 177: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 201: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 225: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 249: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 273: datetime.now() - should use now_utc() - timestamp=datetime.now(),
- Line 396: datetime.now() - should use now_utc() - timestamp=datetime.now() - timedelta(minutes=i),

### tests/unit/test_signal_generation.py
- Line 256: pd.Timestamp.now() - should use now_utc() - 'timestamp': pd.Timestamp.now(),

### tests/unit/test_signal_generation_engine.py
- Line 27: pd.Timestamp.now() - should use now_utc() - dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')

### tests/validate_production_data.py
- Line 287: datetime.now() - should use now_utc() - report_path = f"test_data/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

## Recommended Fixes

1. Add import at the top of each file:
   ```python
   from src.utils.timezone_utils import now_utc, make_aware
   ```

2. Replace datetime calls:
   - `datetime.now()` → `now_utc()`
   - `datetime.utcnow()` → `now_utc()`
   - `datetime.today()` → `now_utc().date()`
   - `pd.Timestamp.now()` → `now_utc()`
   - `pd.Timestamp.utcnow()` → `now_utc()`

3. For existing datetime objects, make them timezone-aware:
   ```python
   dt = make_aware(dt)  # Makes naive datetime UTC-aware
   ```

4. When parsing datetime strings:
   ```python
   from src.utils.timezone_utils import parse_datetime
   dt = parse_datetime(date_string)
   ```
