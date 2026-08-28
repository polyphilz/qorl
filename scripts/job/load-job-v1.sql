\set ON_ERROR_STOP on
\timing on

SET application_name = 'qorl-job-v1-loader';
SET statement_timeout = 0;

\ir /qorl/job-source/schema.sql

\copy public.aka_name FROM '/qorl/job-data/aka_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.aka_title FROM '/qorl/job-data/aka_title.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.cast_info FROM '/qorl/job-data/cast_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.char_name FROM '/qorl/job-data/char_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.comp_cast_type FROM '/qorl/job-data/comp_cast_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_name FROM '/qorl/job-data/company_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_type FROM '/qorl/job-data/company_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.complete_cast FROM '/qorl/job-data/complete_cast.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.info_type FROM '/qorl/job-data/info_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.keyword FROM '/qorl/job-data/keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.kind_type FROM '/qorl/job-data/kind_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.link_type FROM '/qorl/job-data/link_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_companies FROM '/qorl/job-data/movie_companies.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info FROM '/qorl/job-data/movie_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info_idx FROM '/qorl/job-data/movie_info_idx.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_keyword FROM '/qorl/job-data/movie_keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_link FROM '/qorl/job-data/movie_link.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.name FROM '/qorl/job-data/name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.person_info FROM '/qorl/job-data/person_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.role_type FROM '/qorl/job-data/role_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.title FROM '/qorl/job-data/title.csv' WITH (FORMAT csv, ESCAPE E'\\')

\ir /qorl/job-source/fkindexes.sql

GRANT SELECT ON ALL TABLES IN SCHEMA public TO qorl_runner;
