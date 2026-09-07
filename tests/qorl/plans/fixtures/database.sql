-- Synthetic fixture from the PlanAction audit: repeated keys make memoization viable.
CREATE TABLE audit_a AS
SELECT i AS id, i % 100 AS k, repeat('a', 20) AS payload FROM generate_series(1, 20000) i;
CREATE TABLE audit_b AS
SELECT i AS id, i % 100 AS k, repeat('b', 20) AS payload FROM generate_series(1, 2000) i;
CREATE TABLE audit_c AS
SELECT i AS id, i % 100 AS k, repeat('c', 20) AS payload FROM generate_series(1, 1000) i;
CREATE INDEX audit_a_id_idx ON audit_a(id);
CREATE INDEX audit_a_k_idx ON audit_a(k);
CREATE INDEX audit_b_id_idx ON audit_b(id);
CREATE INDEX audit_b_k_idx ON audit_b(k);
CREATE INDEX audit_c_id_idx ON audit_c(id);
CREATE INDEX audit_c_k_idx ON audit_c(k);
VACUUM ANALYZE audit_a;
VACUUM ANALYZE audit_b;
VACUUM ANALYZE audit_c;

-- Session defaults for the two-table memoization reproduction. Positive method
-- hints override these; none of these tests is a latency benchmark.
ALTER DATABASE qorl SET enable_hashjoin = off;
ALTER DATABASE qorl SET enable_mergejoin = off;
