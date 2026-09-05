WITH expected_tables(table_name) AS (
    SELECT unnest(ARRAY[{table_array}]::text[])
), row_counts AS (
    {row_queries}
), columns AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', c.table_name,
            'ordinal', c.ordinal_position,
            'column', c.column_name,
            'data_type', c.data_type,
            'udt_name', c.udt_name,
            'nullable', c.is_nullable,
            'default', c.column_default,
            'character_maximum_length', c.character_maximum_length,
            'numeric_precision', c.numeric_precision,
            'numeric_scale', c.numeric_scale
        ) ORDER BY c.table_name, c.ordinal_position
    ) AS value
    FROM information_schema.columns AS c
    JOIN expected_tables AS e USING (table_name)
    WHERE c.table_schema = 'public'
), constraints AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'name', constraint_row.conname,
            'type', constraint_row.contype,
            'definition', pg_get_constraintdef(constraint_row.oid, true),
            'validated', constraint_row.convalidated
        ) ORDER BY table_class.relname, constraint_row.conname
    ) AS value
    FROM pg_constraint AS constraint_row
    JOIN pg_class AS table_class ON table_class.oid = constraint_row.conrelid
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
), indexes AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'name', index_class.relname,
            'definition', pg_get_indexdef(index_row.indexrelid),
            'primary', index_row.indisprimary,
            'unique', index_row.indisunique,
            'valid', index_row.indisvalid,
            'ready', index_row.indisready
        ) ORDER BY table_class.relname, index_class.relname
    ) AS value
    FROM pg_index AS index_row
    JOIN pg_class AS table_class ON table_class.oid = index_row.indrelid
    JOIN pg_class AS index_class ON index_class.oid = index_row.indexrelid
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
), statistics AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', stats.tablename,
            'column', stats.attname,
            'inherited', stats.inherited,
            'null_frac', stats.null_frac,
            'avg_width', stats.avg_width,
            'n_distinct', stats.n_distinct,
            'most_common_vals', stats.most_common_vals::text,
            'most_common_freqs', stats.most_common_freqs::text,
            'histogram_bounds', stats.histogram_bounds::text,
            'correlation', stats.correlation,
            'most_common_elems', stats.most_common_elems::text,
            'most_common_elem_freqs', stats.most_common_elem_freqs::text,
            'elem_count_histogram', stats.elem_count_histogram::text
        ) ORDER BY stats.tablename, stats.attname, stats.inherited
    ) AS value
    FROM pg_stats AS stats
    JOIN expected_tables AS e ON e.table_name = stats.tablename
    WHERE stats.schemaname = 'public'
), relations AS (
    SELECT jsonb_agg(
        jsonb_build_object(
            'table', table_class.relname,
            'relpages', table_class.relpages,
            'reltuples', table_class.reltuples,
            'relallvisible', table_class.relallvisible,
            'relfrozenxid', table_class.relfrozenxid::text,
            'frozen_xid_age', age(table_class.relfrozenxid),
            'relation_bytes', pg_relation_size(table_class.oid),
            'total_relation_bytes', pg_total_relation_size(table_class.oid)
        ) ORDER BY table_class.relname
    ) AS value
    FROM pg_class AS table_class
    JOIN pg_namespace AS namespace_row ON namespace_row.oid = table_class.relnamespace
    JOIN expected_tables AS e ON e.table_name = table_class.relname
    WHERE namespace_row.nspname = 'public'
)
SELECT jsonb_build_object(
    'identity', jsonb_build_object(
        'server_version_num', current_setting('server_version_num'),
        'database', current_database(),
        'encoding', pg_encoding_to_char(database_row.encoding),
        'collation', database_row.datcollate,
        'ctype', database_row.datctype,
        'system_identifier', (SELECT system_identifier::text FROM pg_control_system()),
        'pg_hint_plan_version', (
            SELECT extversion FROM pg_extension WHERE extname = 'pg_hint_plan'
        )
    ),
    'table_names', (
        SELECT jsonb_agg(class_row.relname ORDER BY class_row.relname)
        FROM pg_class AS class_row
        JOIN pg_namespace AS namespace_row ON namespace_row.oid = class_row.relnamespace
        WHERE namespace_row.nspname = 'public'
          AND class_row.relkind IN ('r', 'p')
    ),
    'table_rows', (
        SELECT jsonb_object_agg(table_name, row_count ORDER BY table_name)
        FROM row_counts
    ),
    'columns', (SELECT value FROM columns),
    'constraints', (SELECT value FROM constraints),
    'indexes', (SELECT value FROM indexes),
    'statistics', (SELECT value FROM statistics),
    'relations', (SELECT value FROM relations)
)
FROM pg_database AS database_row
WHERE database_row.datname = current_database();
