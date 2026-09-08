"""Read-only public-schema metadata, bounded for agent observations."""

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from qorl.postgres.exceptions import PostgresError
from qorl.postgres.schemas import (
    RelationMetadata,
    RelationStatistics,
)

METADATA_ITEMS = 64
DISTRIBUTION_ITEMS = 16
VALUE_BYTES = 128
RELATION_BYTES = 32_768


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def read_record[T: BaseModel](
    execute: Callable[[str], str], sql: str, record: type[T]
) -> T:
    try:
        return record.model_validate_json(execute(sql).strip())
    except ValidationError as error:
        raise PostgresError(
            "PostgreSQL returned invalid inspection metadata"
        ) from error


def inspect_relation(execute: Callable[[str], str], table: str) -> RelationMetadata:
    name = literal(f"public.{table}")
    result = read_record(
        execute,
        f"""
WITH relation AS (SELECT oid, reltuples FROM pg_class WHERE oid = to_regclass({name})),
columns AS (
 SELECT attname AS name, format_type(atttypid, atttypmod) AS type, NOT attnotnull AS nullable
 FROM pg_attribute WHERE attrelid = to_regclass({name}) AND attnum > 0 AND NOT attisdropped ORDER BY attnum
), indexes AS (
 SELECT indexname AS name, indexdef AS definition FROM pg_indexes
 WHERE schemaname = 'public' AND tablename = {literal(table)} ORDER BY indexname
), extended AS (
 SELECT statistics_name AS name, attnames AS columns, kinds FROM pg_stats_ext
 WHERE schemaname = 'public' AND tablename = {literal(table)} ORDER BY statistics_name
)
SELECT json_build_object(
 'exists', EXISTS(SELECT 1 FROM relation),
 'estimated_rows', (SELECT reltuples FROM relation),
 'table_bytes', (SELECT pg_table_size(oid) FROM relation),
 'indexes_bytes', (SELECT pg_indexes_size(oid) FROM relation),
 'total_bytes', (SELECT pg_total_relation_size(oid) FROM relation),
 'columns', (SELECT COALESCE(json_agg(c), '[]'::json) FROM (SELECT * FROM columns LIMIT {METADATA_ITEMS}) c),
 'indexes', (SELECT COALESCE(json_agg(i), '[]'::json) FROM (SELECT * FROM indexes LIMIT {METADATA_ITEMS}) i),
 'extended_statistics', (SELECT COALESCE(json_agg(e), '[]'::json) FROM (SELECT * FROM extended LIMIT {METADATA_ITEMS}) e),
 'omitted_columns', GREATEST(0, (SELECT count(*) FROM columns) - {METADATA_ITEMS}),
 'omitted_indexes', GREATEST(0, (SELECT count(*) FROM indexes) - {METADATA_ITEMS}),
 'omitted_extended_statistics', GREATEST(0, (SELECT count(*) FROM extended) - {METADATA_ITEMS})
);
""",
        RelationMetadata,
    )
    lists = ("columns", "indexes", "extended_statistics")
    while len(json.dumps(result.model_dump(mode="json")).encode()) > RELATION_BYTES:
        largest = max(
            lists, key=lambda name: len(json.dumps(result.model_dump()[name]).encode())
        )
        if largest == "columns":
            result.columns.pop()
            result.omitted_columns += 1
        elif largest == "indexes":
            result.indexes.pop()
            result.omitted_indexes += 1
        else:
            result.extended_statistics.pop()
            result.omitted_extended_statistics += 1
    return result


def column_statistics(
    execute: Callable[[str], str], table: str, columns: list[str]
) -> RelationStatistics:
    requested = ",".join(
        f"({literal(column)}, {index})" for index, column in enumerate(columns)
    )
    result = read_record(
        execute,
        f"""
WITH requested(name, position) AS (VALUES {requested})
SELECT json_build_object('columns', json_agg(json_build_object(
 'column', r.name,
 'status', CASE WHEN a.attname IS NULL THEN 'missing_column' WHEN s.attname IS NULL THEN 'missing_statistics' ELSE 'available' END,
 'null_fraction', s.null_frac, 'average_width_bytes', s.avg_width,
 'n_distinct', s.n_distinct, 'correlation', s.correlation,
 'most_common_values', jsonb_path_query_array(to_jsonb(s.most_common_vals), '$[0 to {DISTRIBUTION_ITEMS - 1}]'),
 'most_common_frequencies', s.most_common_freqs[1:{DISTRIBUTION_ITEMS}],
 'histogram_bounds', to_jsonb(s.histogram_bounds),
 'common_value_count', COALESCE(array_length(s.most_common_vals, 1), 0),
 'histogram_bound_count', COALESCE(array_length(s.histogram_bounds, 1), 0)
) ORDER BY r.position))
FROM requested r
LEFT JOIN pg_attribute a ON a.attrelid = to_regclass({literal("public." + table)}) AND a.attname = r.name AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_stats s ON s.schemaname = 'public' AND s.tablename = {literal(table)} AND s.attname = a.attname AND NOT s.inherited;
""",
        RelationStatistics,
    )
    for item in result.columns:
        bounds = item.histogram_bounds
        if bounds is not None:
            count = len(bounds)
            kept = min(count, DISTRIBUTION_ITEMS)
            positions = (
                [index * (count - 1) // (kept - 1) for index in range(kept)]
                if kept > 1
                else list(range(kept))
            )
            item.histogram_bound_positions = positions
            item.histogram_bounds = [bounds[index] for index in positions]
        item.omitted_common_values = max(
            0, item.common_value_count - DISTRIBUTION_ITEMS
        )
        item.omitted_histogram_bounds = max(
            0, item.histogram_bound_count - len(item.histogram_bound_positions)
        )
        for values, omitted in (
            (item.most_common_values, item.omitted_common_value_indexes),
            (item.histogram_bounds, item.omitted_histogram_bound_indexes),
        ):
            for index, value in enumerate(values or []):
                if len(json.dumps(value).encode()) > VALUE_BYTES:
                    assert values is not None
                    values[index] = None
                    omitted.append(index)
    return result
