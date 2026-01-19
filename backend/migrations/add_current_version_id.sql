-- Migration: Add current_version_id column to configurations table
-- This links the current configuration to the ConfigVersion it's based on
-- Date: 2024
-- 
-- Note: SQLite doesn't support ALTER TABLE ADD COLUMN with REFERENCES constraint directly.
-- The foreign key constraint will be enforced by SQLAlchemy/application layer.
-- For PostgreSQL, you can use: ALTER TABLE configurations ADD COLUMN current_version_id INTEGER REFERENCES config_versions(id);

-- SQLite-compatible: Add the column (nullable initially to support existing data)
ALTER TABLE configurations ADD COLUMN current_version_id INTEGER;

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS ix_configurations_current_version_id ON configurations(current_version_id);

-- Note: Existing configurations will have current_version_id = NULL
-- This is expected - they will show "vN/A" until a rollback or snapshot links them to a version
-- The foreign key relationship is maintained by SQLAlchemy models, not database constraints in SQLite
