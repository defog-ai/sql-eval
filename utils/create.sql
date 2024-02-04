-- Lists the SQL commands to create the tables used for storing run data
-- The database is called "sql_eval", and the tables are "prompt" and "eval"

CREATE TABLE IF NOT EXISTS eval (
  -- first, metadata about the run
  run_id VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  runner_type VARCHAR(255),
  prompt_id VARCHAR(255),
  model VARCHAR(255),
  num_beams INT,
  db_type VARCHAR(255),
  gpu_name VARCHAR(255),
  gpu_memory INT,
  gpu_driver_version VARCHAR(255),
  gpu_cuda_version VARCHAR(255),

  -- then, data about actual questions
  question TEXT,
  golden_query TEXT,
  db_name VARCHAR(255),
  query_category VARCHAR(255),
  generated_query TEXT,
  error_msg TEXT,
  exact_match BOOLEAN,
  correct BOOLEAN,
  error_db_exec BOOLEAN,
  latency_seconds FLOAT,
  tokens_used INT
);

-- indexes for the table on run_id, model, db_name, and query_category
CREATE INDEX IF NOT EXISTS eval_run_id ON eval(run_id);
CREATE INDEX IF NOT EXISTS eval_model ON eval(model);
CREATE INDEX IF NOT EXISTS eval_db_name ON eval(db_name);
CREATE INDEX IF NOT EXISTS eval_query_category ON eval(query_category);

-- create prompt table
CREATE TABLE IF NOT EXISTS prompt (
  prompt_id VARCHAR(255) PRIMARY KEY,
  prompt TEXT
);