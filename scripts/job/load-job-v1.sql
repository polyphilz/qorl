\set ON_ERROR_STOP on
\timing on

SET application_name = 'qprl-job-v1-loader';
SET statement_timeout = 0;

\ir /qprl/job-source/schema.sql

\copy public.aka_name FROM '/qprl/job-data/aka_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.aka_title FROM '/qprl/job-data/aka_title.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.cast_info FROM '/qprl/job-data/cast_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.char_name FROM '/qprl/job-data/char_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.comp_cast_type FROM '/qprl/job-data/comp_cast_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_name FROM '/qprl/job-data/company_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_type FROM '/qprl/job-data/company_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.complete_cast FROM '/qprl/job-data/complete_cast.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.info_type FROM '/qprl/job-data/info_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.keyword FROM '/qprl/job-data/keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.kind_type FROM '/qprl/job-data/kind_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.link_type FROM '/qprl/job-data/link_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_companies FROM '/qprl/job-data/movie_companies.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info FROM '/qprl/job-data/movie_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info_idx FROM '/qprl/job-data/movie_info_idx.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_keyword FROM '/qprl/job-data/movie_keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_link FROM '/qprl/job-data/movie_link.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.name FROM '/qprl/job-data/name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.person_info FROM '/qprl/job-data/person_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.role_type FROM '/qprl/job-data/role_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.title FROM '/qprl/job-data/title.csv' WITH (FORMAT csv, ESCAPE E'\\')

\ir /qprl/job-source/fkindexes.sql

GRANT SELECT ON ALL TABLES IN SCHEMA public TO qp_agent;
