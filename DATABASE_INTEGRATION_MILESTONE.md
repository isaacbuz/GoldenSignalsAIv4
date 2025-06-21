# Database Integration Milestone for Production

## Overview
Currently, GoldenSignalsAI trains ML models directly from API data without persistent storage. For production readiness, we need comprehensive database integration for data storage, model versioning, and performance tracking.

## Current State vs Production Requirements

### Current State ❌
- **Data Fetching**: Direct from Yahoo Finance API during training
- **Storage**: In-memory only during training
- **Models**: Saved as pickle files
- **History**: No historical data preservation
- **Tracking**: No model performance tracking

### Production Requirements ✅
- **Data Lake**: Store all historical market data
- **Feature Store**: Pre-computed features for fast training
- **Model Registry**: Version control for models
- **Performance DB**: Track model performance over time
- **Audit Trail**: Complete training history

## Database Architecture

### 1. **Time-Series Database (Primary)**
**Technology**: TimescaleDB (PostgreSQL extension) or InfluxDB

**Purpose**: Store market data efficiently
```sql
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (time, symbol)
);

-- Create hypertable for time-series optimization
SELECT create_hypertable('market_data', 'time');
```

### 2. **Feature Store**
**Technology**: PostgreSQL with materialized views

**Purpose**: Pre-computed features ready for training
```sql
CREATE TABLE feature_store (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_set JSONB,
    version INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feature_store_symbol_time 
ON feature_store(symbol, timestamp);
```

### 3. **Model Registry**
**Technology**: PostgreSQL + S3/MinIO for model artifacts

**Schema**:
```sql
CREATE TABLE model_registry (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    training_date TIMESTAMPTZ NOT NULL,
    metrics JSONB,
    hyperparameters JSONB,
    feature_columns TEXT[],
    artifact_path TEXT,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_id UUID REFERENCES model_registry(model_id),
    evaluation_date TIMESTAMPTZ NOT NULL,
    symbol TEXT,
    predictions JSONB,
    actual_values JSONB,
    metrics JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 4. **Training Metadata**
```sql
CREATE TABLE training_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status TEXT NOT NULL,
    data_range DATERANGE,
    symbols TEXT[],
    total_samples BIGINT,
    training_params JSONB,
    error_log TEXT,
    created_by TEXT
);
```

## Implementation Phases

### Phase 1: Data Collection Infrastructure (Week 1-2)
- [ ] Set up TimescaleDB for market data
- [ ] Create data ingestion pipeline
- [ ] Implement data validation
- [ ] Set up automated backfill for 20 years
- [ ] Create data quality monitoring

### Phase 2: Feature Engineering Pipeline (Week 3)
- [ ] Design feature store schema
- [ ] Implement feature calculation jobs
- [ ] Create feature versioning system
- [ ] Set up incremental updates
- [ ] Add feature importance tracking

### Phase 3: Model Registry & Versioning (Week 4)
- [ ] Implement model registry database
- [ ] Create model artifact storage (S3/MinIO)
- [ ] Build model deployment pipeline
- [ ] Add A/B testing infrastructure
- [ ] Create rollback mechanisms

### Phase 4: Performance Tracking (Week 5)
- [ ] Design performance metrics schema
- [ ] Implement real-time prediction logging
- [ ] Create performance dashboards
- [ ] Set up drift detection
- [ ] Add alerting for model degradation

### Phase 5: Production Training Pipeline (Week 6)
- [ ] Refactor training to use database
- [ ] Implement distributed training
- [ ] Add hyperparameter tracking
- [ ] Create automated retraining
- [ ] Set up experiment tracking (MLflow)

## Database Sizing Estimates

### Market Data (20 years)
- **Symbols**: 30 core + 1000 extended = 1030 symbols
- **Records per symbol**: 5,000 trading days
- **Total records**: ~5.15 million
- **Record size**: ~100 bytes
- **Total size**: ~515 MB (raw)
- **With indexes**: ~1.5 GB

### Feature Store
- **Features per record**: 50
- **Feature record size**: ~500 bytes
- **Total size**: ~2.5 GB
- **With materialized views**: ~5 GB

### Model Registry
- **Models**: 6 types × 100 versions = 600 models
- **Model size**: ~50 MB average
- **Total artifact storage**: ~30 GB

### Performance Data (1 year)
- **Predictions per day**: 1000
- **Record size**: ~1 KB
- **Annual size**: ~365 MB

**Total Database Size**: ~40-50 GB for first year

## Technology Stack

### Primary Database
```yaml
production:
  timeseries_db:
    type: TimescaleDB
    version: 2.x
    specs:
      cpu: 8 cores
      memory: 32 GB
      storage: 500 GB SSD
  
  application_db:
    type: PostgreSQL
    version: 14+
    specs:
      cpu: 4 cores
      memory: 16 GB
      storage: 200 GB SSD
```

### Cache Layer
```yaml
cache:
  redis:
    version: 7.x
    mode: cluster
    memory: 16 GB
    persistence: AOF
```

### Object Storage
```yaml
storage:
  s3_compatible:
    provider: AWS S3 / MinIO
    buckets:
      - goldensignals-models
      - goldensignals-datasets
      - goldensignals-backups
```

## Migration Strategy

### Step 1: Parallel Implementation
1. Keep current file-based system running
2. Build database infrastructure in parallel
3. Start dual-writing to both systems
4. Validate data consistency

### Step 2: Historical Data Migration
```python
# Migration script example
def migrate_historical_data():
    """Migrate 20 years of data to TimescaleDB"""
    
    # 1. Fetch from Yahoo Finance in batches
    for symbol in symbols:
        data = yf.download(symbol, start='2004-01-01')
        
        # 2. Transform to database format
        records = transform_to_timeseries(data)
        
        # 3. Bulk insert with conflict handling
        insert_market_data(records, on_conflict='ignore')
        
        # 4. Calculate and store features
        features = calculate_features(data)
        insert_features(features)
```

### Step 3: Training Pipeline Migration
```python
class DatabaseTrainingPipeline:
    def __init__(self, db_config):
        self.db = TimescaleDBConnection(db_config)
        self.feature_store = FeatureStore(db_config)
        self.model_registry = ModelRegistry(db_config)
    
    def load_training_data(self, start_date, end_date):
        """Load from database instead of API"""
        # Fast query from pre-computed features
        return self.feature_store.get_features(
            symbols=self.symbols,
            start=start_date,
            end=end_date
        )
    
    def save_model(self, model, metrics):
        """Save to model registry"""
        model_id = self.model_registry.register(
            model=model,
            metrics=metrics,
            artifact_path=self.upload_to_s3(model)
        )
        return model_id
```

## Monitoring & Maintenance

### Data Quality Checks
```sql
-- Daily data completeness check
CREATE OR REPLACE FUNCTION check_data_completeness()
RETURNS TABLE(symbol TEXT, missing_dates DATE[])
AS $$
BEGIN
    -- Check for gaps in daily data
    RETURN QUERY
    SELECT 
        symbol,
        array_agg(missing_date)
    FROM (
        SELECT 
            symbol,
            generate_series(
                MIN(time)::date,
                MAX(time)::date,
                '1 day'::interval
            )::date AS expected_date
        FROM market_data
        GROUP BY symbol
    ) expected
    LEFT JOIN market_data actual
    ON expected.symbol = actual.symbol
    AND expected.expected_date = actual.time::date
    WHERE actual.time IS NULL
    GROUP BY symbol;
END;
$$ LANGUAGE plpgsql;
```

### Performance Monitoring
```python
# Track model performance degradation
def monitor_model_drift():
    """Check for model performance drift"""
    current_metrics = db.query("""
        SELECT 
            model_id,
            AVG(metrics->>'accuracy') as avg_accuracy,
            STDDEV(metrics->>'accuracy') as std_accuracy
        FROM model_performance
        WHERE evaluation_date > NOW() - INTERVAL '7 days'
        GROUP BY model_id
    """)
    
    for model in current_metrics:
        if model.avg_accuracy < baseline_accuracy - 2 * std:
            alert_model_degradation(model.model_id)
```

## Cost Estimates

### AWS/Cloud Costs (Monthly)
- **RDS PostgreSQL (db.r6g.xlarge)**: $400
- **TimescaleDB License**: $500
- **S3 Storage (100GB)**: $25
- **ElastiCache Redis**: $200
- **Data Transfer**: $50
- **Backups**: $50

**Total**: ~$1,225/month

### On-Premise Alternative
- **Hardware**: $10,000 (one-time)
- **TimescaleDB License**: $500/month
- **Maintenance**: $500/month

**Total**: $1,000/month after initial investment

## Success Metrics

### Performance KPIs
- ✅ Training time reduced by 80% (data pre-cached)
- ✅ Model deployment time < 5 minutes
- ✅ 99.9% data availability
- ✅ < 100ms feature query latency
- ✅ Automated retraining on schedule

### Business Impact
- ✅ Reproducible model training
- ✅ Complete audit trail for compliance
- ✅ A/B testing for model improvements
- ✅ Real-time performance monitoring
- ✅ Reduced API costs (no repeated fetching)

## Next Steps

1. **Immediate Actions**:
   - [ ] Set up development database environment
   - [ ] Create database schemas
   - [ ] Build data ingestion prototype
   - [ ] Test with 1 year of data

2. **Pilot Program**:
   - [ ] Select 5 symbols for pilot
   - [ ] Implement full pipeline for pilot
   - [ ] Measure performance improvements
   - [ ] Gather feedback

3. **Full Rollout**:
   - [ ] Scale to all symbols
   - [ ] Migrate all models
   - [ ] Deploy monitoring
   - [ ] Train team on new system

## Conclusion

Database integration is critical for GoldenSignalsAI to operate at production scale. This milestone will transform the system from a prototype to a production-ready platform capable of handling institutional trading requirements. 