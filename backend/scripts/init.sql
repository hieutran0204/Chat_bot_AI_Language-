-- name: init.sql
-- description: Initial SQL script run on first PostgreSQL startup — enables pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
