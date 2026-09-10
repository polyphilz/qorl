# CEB selection audit — experiment 025

Audited on 2026-09-10. Recommendation: retain all 300 selected training inputs. No database queries or teacher generation were run for this audit.

## Selection and verification

- Seed 42; 30 tasks from each of the original ten training templates. Exclusion happens before sampling, so all 300 slots are filled.
- Frozen exclusion input contains all 100 original training queries, including any whose earlier generation failed. No new training task ID overlaps that input.
- Validation is exactly the original 20 IDs (10 each from CEB 5a/8a); JOB is exactly the original 113 test IDs. Current split validation confirms no cross-split topology overlap.
- Static parser audit covered 533 distinct SQL files: new training 300, validation 20, original training 100, JOB test 113. All parse as a single SELECT, match catalog checksums, relation mappings and equality-join metadata, reference declared aliases, and have connected join graphs.
- No duplicate raw SQL hashes or parsed/formatted SQL hashes across those 533 files. Normalization used pglast v8.4 RawStream and removes formatting/comments. This does not establish semantic inequivalence on the IMDB data.
- Ten GPT-6 Astra reviewers at medium reasoning each reviewed five complete SQL inputs plus catalog metadata and original examples. The final selected IDs match the preview they reviewed.
- Samples are positions 0, 7, 14, 21 and 29 after sorting each template's 30 queries by `(SQL byte length, task_id)`. This spans text length; it is not random sampling or a runtime-difficulty ranking. All fifty recommendations were keep.
- Focused CLI, experiment-creation and task-selection tests: 131 passed. Changed-file Pyright: zero errors/warnings. Ruff checks/format clean. Config and preparation-stage validation passed without starting generation.

## Findings to carry into generation

Templates 6a/7a retain the benchmark's existing regex-plus-float-cast expressions. Their numeric regex permits an empty string, and AND is not a procedural cast guard. This is a conditional execution risk already present in the original inputs, not an observed failure in this batch. Preserve the fixed SQL; diagnose any execution errors separately from teacher mistakes.

Some inputs have broad substring/list predicates, narrow text conjunctions or substantial overlap with earlier predicate values. In particular, new 7a67 shares many filters with old 7a47 but changes movie-info values, the movie-info type list and person-info type. These are expected within-template variants. No static evidence justified replacing them. Result cardinality, measurement cost and possible speedups remain unknown.

This audit cannot select for successful Leading use, recovery, default fallback or correct final choices before conversations exist. Post-generation review should keep diverse accepted interventions and useful corrections, mask invalid replies while retaining context, flag selection of known-slower/timed-out candidates, account for noisy near ties, and measure student-rendered lengths. Do not require every retained trajectory to produce a novel plan or speedup.

## Per-template findings

| Template | Selected / reviewed | Review |
| --- | ---: | --- |
| 1a | 30 / 5 | 9 aliases, repeated movie/info-type tables. Category lists, years, roles and NULL handling vary; inspect long location lists in 1a2621. |
| 2a | 30 / 5 | 11 aliases. Text, category and NULL predicates vary. 2a51/2a266 have narrow-looking text conjunctions; 2a227 has long location lists. Empty matches are unverified. |
| 3a | 30 / 5 | 10 aliases. Keyword, country, role and year filters vary; 3a484 has the longest keyword payload. No runtime difficulty follows from SQL length. |
| 4a | 30 / 5 | 6 aliases. Phonetic-code, role and note lists plus NULL handling vary; 4a221 has broad lists. This is the smallest selected topology. |
| 6a | 30 / 5 | 14 aliases, repeated tables, numeric ranges and opaque info types. Retain the inherited regex/float-cast flag described above. |
| 7a | 30 / 5 | 16 aliases. Varies numeric ranges, categories and NULL handling. Retain the inherited cast flag; 7a67 overlaps several old 7a47 predicates but changes three predicate groups. |
| 9a | 30 / 5 | 9 aliases with grouped output and substring filters. The 6fd0d124... sample includes ILIKE '%%', which accepts any non-NULL text; other filters and joins still apply. |
| 10a | 30 / 5 | 7 aliases with grouped output. Information-value lists and name substrings vary; type IDs require grounding. Sample role-family variety is limited. |
| 11a | 30 / 5 | 10 aliases with grouping. Role lists, year ranges and substrings vary. Short substrings and broad lists should be interpreted using observed feedback. |
| 11b | 30 / 5 | 12 aliases with grouping and two info-type roles. Varies categories, years and movie/person substrings; broad-list samples are retained. |

## The fifty reviewed inputs

| Template | Task / SQL | Bytes |
| --- | --- | ---: |
| 1a | [ceb-1a-1a2250](../../benchmarks/ceb/queries/1a/1a2250.sql) | 683 |
| 1a | [ceb-1a-1a62](../../benchmarks/ceb/queries/1a/1a62.sql) | 741 |
| 1a | [ceb-1a-1a2296](../../benchmarks/ceb/queries/1a/1a2296.sql) | 767 |
| 1a | [ceb-1a-1a2675](../../benchmarks/ceb/queries/1a/1a2675.sql) | 804 |
| 1a | [ceb-1a-1a2621](../../benchmarks/ceb/queries/1a/1a2621.sql) | 1443 |
| 2a | [ceb-2a-2a51](../../benchmarks/ceb/queries/2a/2a51.sql) | 773 |
| 2a | [ceb-2a-2a123](../../benchmarks/ceb/queries/2a/2a123.sql) | 847 |
| 2a | [ceb-2a-2a429](../../benchmarks/ceb/queries/2a/2a429.sql) | 880 |
| 2a | [ceb-2a-2a266](../../benchmarks/ceb/queries/2a/2a266.sql) | 974 |
| 2a | [ceb-2a-2a227](../../benchmarks/ceb/queries/2a/2a227.sql) | 1371 |
| 3a | [ceb-3a-3a145](../../benchmarks/ceb/queries/3a/3a145.sql) | 772 |
| 3a | [ceb-3a-3a1177](../../benchmarks/ceb/queries/3a/3a1177.sql) | 835 |
| 3a | [ceb-3a-3a419](../../benchmarks/ceb/queries/3a/3a419.sql) | 852 |
| 3a | [ceb-3a-3a89](../../benchmarks/ceb/queries/3a/3a89.sql) | 904 |
| 3a | [ceb-3a-3a484](../../benchmarks/ceb/queries/3a/3a484.sql) | 1157 |
| 4a | [ceb-4a-4a229](../../benchmarks/ceb/queries/4a/4a229.sql) | 506 |
| 4a | [ceb-4a-4a251](../../benchmarks/ceb/queries/4a/4a251.sql) | 596 |
| 4a | [ceb-4a-4a443](../../benchmarks/ceb/queries/4a/4a443.sql) | 694 |
| 4a | [ceb-4a-4a502](../../benchmarks/ceb/queries/4a/4a502.sql) | 824 |
| 4a | [ceb-4a-4a221](../../benchmarks/ceb/queries/4a/4a221.sql) | 1059 |
| 6a | [ceb-6a-6a343](../../benchmarks/ceb/queries/6a/6a343.sql) | 1439 |
| 6a | [ceb-6a-6a80](../../benchmarks/ceb/queries/6a/6a80.sql) | 1502 |
| 6a | [ceb-6a-6a382](../../benchmarks/ceb/queries/6a/6a382.sql) | 1544 |
| 6a | [ceb-6a-6a196](../../benchmarks/ceb/queries/6a/6a196.sql) | 1593 |
| 6a | [ceb-6a-6a518](../../benchmarks/ceb/queries/6a/6a518.sql) | 1805 |
| 7a | [ceb-7a-7a67](../../benchmarks/ceb/queries/7a/7a67.sql) | 1353 |
| 7a | [ceb-7a-7a54](../../benchmarks/ceb/queries/7a/7a54.sql) | 1453 |
| 7a | [ceb-7a-7a16](../../benchmarks/ceb/queries/7a/7a16.sql) | 1510 |
| 7a | [ceb-7a-7a96](../../benchmarks/ceb/queries/7a/7a96.sql) | 1553 |
| 7a | [ceb-7a-7a87](../../benchmarks/ceb/queries/7a/7a87.sql) | 1738 |
| 9a | [ceb-9a-6fd0d12433f371879163b54eadde49a27008a1a3](../../benchmarks/ceb/queries/9a/6fd0d12433f371879163b54eadde49a27008a1a3.sql) | 696 |
| 9a | [ceb-9a-8c245b6ec813c34c5baf028892d71426039a912a](../../benchmarks/ceb/queries/9a/8c245b6ec813c34c5baf028892d71426039a912a.sql) | 748 |
| 9a | [ceb-9a-1c8b435acaf6ca76b7c85372ccc38ae08d9e0dc8](../../benchmarks/ceb/queries/9a/1c8b435acaf6ca76b7c85372ccc38ae08d9e0dc8.sql) | 760 |
| 9a | [ceb-9a-b51c3001e1ca81425d991ccc9ec27bedfc8a2501](../../benchmarks/ceb/queries/9a/b51c3001e1ca81425d991ccc9ec27bedfc8a2501.sql) | 771 |
| 9a | [ceb-9a-6fd7be280a54ef8b0afae1bcb9b352ca96db4f0b](../../benchmarks/ceb/queries/9a/6fd7be280a54ef8b0afae1bcb9b352ca96db4f0b.sql) | 792 |
| 10a | [ceb-10a-c34f57e8d6907c58e46c2d53d7051bef29802cf6](../../benchmarks/ceb/queries/10a/c34f57e8d6907c58e46c2d53d7051bef29802cf6.sql) | 591 |
| 10a | [ceb-10a-a6e65609ef137f2f684344c9ff20efb830c4a2a7](../../benchmarks/ceb/queries/10a/a6e65609ef137f2f684344c9ff20efb830c4a2a7.sql) | 631 |
| 10a | [ceb-10a-feb4a84432cbf3c03659dec9f994e45c20117dce](../../benchmarks/ceb/queries/10a/feb4a84432cbf3c03659dec9f994e45c20117dce.sql) | 650 |
| 10a | [ceb-10a-21b1b4f557d5399444d82d3591c389c78e923a84](../../benchmarks/ceb/queries/10a/21b1b4f557d5399444d82d3591c389c78e923a84.sql) | 707 |
| 10a | [ceb-10a-e728f419a26c4df019387b7cf2fbdfba53a80648](../../benchmarks/ceb/queries/10a/e728f419a26c4df019387b7cf2fbdfba53a80648.sql) | 847 |
| 11a | [ceb-11a-b2dfd0f5734f93f4738d9ad5900f6e632a509ba7](../../benchmarks/ceb/queries/11a/b2dfd0f5734f93f4738d9ad5900f6e632a509ba7.sql) | 793 |
| 11a | [ceb-11a-5ed0c95c39e65449d9dda7875bc0c2ef7311cdb7](../../benchmarks/ceb/queries/11a/5ed0c95c39e65449d9dda7875bc0c2ef7311cdb7.sql) | 816 |
| 11a | [ceb-11a-e260767c0459465c69baf1fbfc54bf00eeb37052](../../benchmarks/ceb/queries/11a/e260767c0459465c69baf1fbfc54bf00eeb37052.sql) | 838 |
| 11a | [ceb-11a-4157947df05d280ffd7636554549fba7e8639bca](../../benchmarks/ceb/queries/11a/4157947df05d280ffd7636554549fba7e8639bca.sql) | 855 |
| 11a | [ceb-11a-3b59b2b066dc2dd7d3850b4bdb6188ff1d1a0e60](../../benchmarks/ceb/queries/11a/3b59b2b066dc2dd7d3850b4bdb6188ff1d1a0e60.sql) | 894 |
| 11b | [ceb-11b-c9ab56353e579e255582baa3fe52cf7e920edfa4](../../benchmarks/ceb/queries/11b/c9ab56353e579e255582baa3fe52cf7e920edfa4.sql) | 979 |
| 11b | [ceb-11b-c07bda4b81f466a0b33225007a3705fc4f9decf3](../../benchmarks/ceb/queries/11b/c07bda4b81f466a0b33225007a3705fc4f9decf3.sql) | 1019 |
| 11b | [ceb-11b-7757b29fd61ff132280698ed891770a20e334c29](../../benchmarks/ceb/queries/11b/7757b29fd61ff132280698ed891770a20e334c29.sql) | 1038 |
| 11b | [ceb-11b-d252c19560dc11373b382236f5545ed31af977ba](../../benchmarks/ceb/queries/11b/d252c19560dc11373b382236f5545ed31af977ba.sql) | 1058 |
| 11b | [ceb-11b-da2a1896fdeaec377f6476aa39463b0a8dd850d5](../../benchmarks/ceb/queries/11b/da2a1896fdeaec377f6476aa39463b0a8dd850d5.sql) | 1102 |

## Frozen selection checksums

| File | SHA-256 |
| --- | --- |
| training-tasks.json | `ecf4bb48b0be90d0a177a25c7a45791f703c9fbf077b10653ebcc42e56cbc42f` |
| validation-tasks.json | `cc8d24f03c553e9d61bb8fcf296f583b3649bcae3ac1dbd80bd7275bd4ef165b` |
| test-tasks.json | `b3784578c30aacb96014dd95d8fbf670293578cdeecae536339752221be6d0e2` |
| excluded-tasks-000.json | `0c72e30bf2e1e10f9424ce95cec086c476665bab7150492abe0a492f6efc9602` |

Full per-template reports and the one-off parser audit are available locally under `outputs/analysis/next-astra-ceb-300/`. No parser dependency was added to the project.
