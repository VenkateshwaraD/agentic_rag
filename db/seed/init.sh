#!/bin/bash
# Compose initdb hook: create ragdb + opsdb, apply ops schema + readonly role.
set -e
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<-SQL
    CREATE DATABASE ragdb;
    CREATE DATABASE opsdb;
SQL
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d opsdb -f /seed/init_ops.sql
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d opsdb -f /seed/readonly_role.sql
