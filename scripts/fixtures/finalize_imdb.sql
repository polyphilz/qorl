\set ON_ERROR_STOP on
\timing on

SET application_name = 'qorl-imdb-finalizer';
SET statement_timeout = 0;

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
