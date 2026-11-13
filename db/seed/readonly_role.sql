-- Read-only role for ops tools. Applied once by `copilot db init` or compose initdb.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_readonly') THEN
        CREATE ROLE rag_readonly LOGIN PASSWORD 'readonly';
    END IF;
END $$;

GRANT CONNECT ON DATABASE opsdb TO rag_readonly;
GRANT USAGE ON SCHEMA public TO rag_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO rag_readonly;
