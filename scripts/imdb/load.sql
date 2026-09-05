\set ON_ERROR_STOP on
\timing on

SET application_name = 'qorl-imdb-loader';
SET statement_timeout = 0;

CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);

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

create index company_id_movie_companies on movie_companies(company_id);
create index company_type_id_movie_companies on movie_companies(company_type_id);
create index info_type_id_movie_info_idx on movie_info_idx(info_type_id);
create index info_type_id_movie_info on movie_info(info_type_id);
create index info_type_id_person_info on person_info(info_type_id);
create index keyword_id_movie_keyword on movie_keyword(keyword_id);
create index kind_id_aka_title on aka_title(kind_id);
create index kind_id_title on title(kind_id);
create index linked_movie_id_movie_link on movie_link(linked_movie_id);
create index link_type_id_movie_link on movie_link(link_type_id);
create index movie_id_aka_title on aka_title(movie_id);
create index movie_id_cast_info on cast_info(movie_id);
create index movie_id_complete_cast on complete_cast(movie_id);
create index movie_id_movie_companies on movie_companies(movie_id);
create index movie_id_movie_info_idx on movie_info_idx(movie_id);
create index movie_id_movie_keyword on movie_keyword(movie_id);
create index movie_id_movie_link on movie_link(movie_id);
create index movie_id_movie_info on movie_info(movie_id);
create index person_id_aka_name on aka_name(person_id);
create index person_id_cast_info on cast_info(person_id);
create index person_id_person_info on person_info(person_id);
create index person_role_id_cast_info on cast_info(person_role_id);
create index role_id_cast_info on cast_info(role_id);

GRANT SELECT ON ALL TABLES IN SCHEMA public TO qorl_runner;

SET application_name = 'qorl-imdb-finalizer';

VACUUM (FREEZE, ANALYZE)
    public.aka_name,
    public.aka_title,
    public.cast_info,
    public.char_name,
    public.comp_cast_type,
    public.company_name,
    public.company_type,
    public.complete_cast,
    public.info_type,
    public.keyword,
    public.kind_type,
    public.link_type,
    public.movie_companies,
    public.movie_info,
    public.movie_info_idx,
    public.movie_keyword,
    public.movie_link,
    public.name,
    public.person_info,
    public.role_type,
    public.title;

CHECKPOINT;
