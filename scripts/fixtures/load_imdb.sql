\set ON_ERROR_STOP on
\timing on

SET application_name = 'qorl-imdb-loader';
SET statement_timeout = 0;

\ir /qorl/imdb-source/schema.sql

\copy public.aka_name FROM '/qorl/imdb-data/aka_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.aka_title FROM '/qorl/imdb-data/aka_title.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.cast_info FROM '/qorl/imdb-data/cast_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.char_name FROM '/qorl/imdb-data/char_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.comp_cast_type FROM '/qorl/imdb-data/comp_cast_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_name FROM '/qorl/imdb-data/company_name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.company_type FROM '/qorl/imdb-data/company_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.complete_cast FROM '/qorl/imdb-data/complete_cast.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.info_type FROM '/qorl/imdb-data/info_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.keyword FROM '/qorl/imdb-data/keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.kind_type FROM '/qorl/imdb-data/kind_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.link_type FROM '/qorl/imdb-data/link_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_companies FROM '/qorl/imdb-data/movie_companies.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info FROM '/qorl/imdb-data/movie_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_info_idx FROM '/qorl/imdb-data/movie_info_idx.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_keyword FROM '/qorl/imdb-data/movie_keyword.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.movie_link FROM '/qorl/imdb-data/movie_link.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.name FROM '/qorl/imdb-data/name.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.person_info FROM '/qorl/imdb-data/person_info.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.role_type FROM '/qorl/imdb-data/role_type.csv' WITH (FORMAT csv, ESCAPE E'\\')
\copy public.title FROM '/qorl/imdb-data/title.csv' WITH (FORMAT csv, ESCAPE E'\\')

\ir /qorl/imdb-source/fkindexes.sql

GRANT SELECT ON ALL TABLES IN SCHEMA public TO qorl_runner;
