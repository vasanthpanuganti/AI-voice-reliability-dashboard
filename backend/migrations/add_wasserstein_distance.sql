-- Migration: Add wasserstein_distance column to drift_metrics table
-- Date: 2026-01-18
-- Description: Adds Wasserstein distance metric for improved embedding drift detection

-- Add wasserstein_distance column to drift_metrics
ALTER TABLE drift_metrics ADD COLUMN IF NOT EXISTS wasserstein_distance FLOAT;

-- Add index for performance (optional but recommended)
CREATE INDEX IF NOT EXISTS idx_drift_metrics_wasserstein
ON drift_metrics(wasserstein_distance)
WHERE wasserstein_distance IS NOT NULL;

-- Verify the column was added
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'drift_metrics'
  AND column_name = 'wasserstein_distance';
