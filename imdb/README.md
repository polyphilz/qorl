# IMDb fixture

JOB and CEB use the same prepared May 2013 IMDb database.

- [build.json](build.json) is the **recipe**: where to download the IMDb rows and
  schema/index definitions, their exact versions and checksums, and how to load them.
- [archive.json](archive.json) is the **record of the prepared database**: the size
  and checksum of `imdb.tar.gz`, its PostgreSQL identity and image, and references to
  the build recipe and verification results. Workers use it to check what they restore.

Both JSON files are tracked in Git. The actual database, `imdb.tar.gz`, stays on
the benchmark host and is ignored by Git, along with `raw/` downloads and
`verification/` evidence.

[Fixture scripts](../scripts/fixtures/README.md) build and verify the archive.
