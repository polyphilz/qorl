# syntax=docker/dockerfile:1

# Pin the multi-platform index. compose.yaml deliberately selects linux/amd64,
# which resolves to the amd64 manifest recorded in versions.json.
ARG POSTGRES_BASE_IMAGE=postgres:18.6-bookworm@sha256:1c59e2c3c818eaa0f0628f695b36e7c9e362d6b219b36a54a32df645cbd7e1af

FROM ${POSTGRES_BASE_IMAGE} AS pg_hint_plan_builder

ARG PG_HINT_PLAN_TAG=REL18_1_8_0
ARG PG_HINT_PLAN_SOURCE_SHA256=c3d8aa1e468b6b8371fb09d971152b9ac1610118fbd0c76c7e308e88a1cc0ef7
ARG POSTGRES_DEV_PACKAGE_VERSION=18.6-1.pgdg12+2

RUN set -eux; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        flex \
        "postgresql-server-dev-18=${POSTGRES_DEV_PACKAGE_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/pg_hint_plan

RUN set -eux; \
    curl --fail --location --show-error \
        --output pg_hint_plan.tar.gz \
        "https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/${PG_HINT_PLAN_TAG}.tar.gz"; \
    echo "${PG_HINT_PLAN_SOURCE_SHA256}  pg_hint_plan.tar.gz" | sha256sum --check; \
    mkdir source; \
    tar --extract --gzip --file pg_hint_plan.tar.gz --directory source --strip-components=1; \
    make --directory source --jobs="$(nproc)"; \
    make --directory source DESTDIR=/opt/pg_hint_plan-install install; \
    mkdir -p /opt/pg_hint_plan-install/usr/share/qorl; \
    dpkg-query --show --showformat='${binary:Package}\t${Version}\n' \
        | LC_ALL=C sort \
        > /opt/pg_hint_plan-install/usr/share/qorl/extension-build-packages.tsv; \
    cd /opt/pg_hint_plan-install/usr/lib/postgresql/18/lib; \
    sha256sum pg_hint_plan.so \
        > /opt/pg_hint_plan-install/usr/share/qorl/pg_hint_plan.so.sha256

FROM ${POSTGRES_BASE_IMAGE}

ARG PG_HINT_PLAN_TAG=REL18_1_8_0
ARG PG_HINT_PLAN_VERSION=1.8.0
ARG PG_HINT_PLAN_COMMIT=88437bde5947e7bc40719e8b82eb8be2b39d71e8
ARG PG_HINT_PLAN_SOURCE_SHA256=c3d8aa1e468b6b8371fb09d971152b9ac1610118fbd0c76c7e308e88a1cc0ef7

LABEL org.opencontainers.image.title="qorl-postgres" \
      org.opencontainers.image.description="Pinned PostgreSQL and pg_hint_plan runtime for QORL plan-steering experiments" \
      org.opencontainers.image.version="18.6-pg_hint_plan-1.8.0" \
      org.opencontainers.image.source="https://github.com/ossc-db/pg_hint_plan" \
      io.qorl.benchmark.config-id="benchmark-v2" \
      io.qorl.postgresql.version="18.6" \
      io.qorl.pg_hint_plan.version="${PG_HINT_PLAN_VERSION}" \
      io.qorl.pg_hint_plan.tag="${PG_HINT_PLAN_TAG}" \
      io.qorl.pg_hint_plan.commit="${PG_HINT_PLAN_COMMIT}" \
      io.qorl.pg_hint_plan.source-sha256="${PG_HINT_PLAN_SOURCE_SHA256}"

ENV PG_HINT_PLAN_VERSION=${PG_HINT_PLAN_VERSION} \
    PG_HINT_PLAN_TAG=${PG_HINT_PLAN_TAG} \
    PG_HINT_PLAN_COMMIT=${PG_HINT_PLAN_COMMIT}

COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/lib/postgresql/18/lib/ /usr/lib/postgresql/18/lib/
COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/share/postgresql/18/extension/ /usr/share/postgresql/18/extension/
COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/share/qorl/ /usr/share/qorl/

COPY contract/benchmark.conf /etc/qorl/benchmark-v2.conf
COPY contract/benchmark.expected.json /usr/share/qorl/benchmark-v2.expected.json
COPY versions.json /usr/share/qorl/versions.json
COPY --chmod=0555 contract/assert-benchmark-config.sh /usr/local/bin/qorl-assert-benchmark-config
COPY --chmod=0555 contract/dump-postgres-state.sh /usr/local/bin/qorl-dump-postgres-state
COPY --chmod=0555 initdb/bootstrap.sh /docker-entrypoint-initdb.d/bootstrap.sh

RUN set -eux; \
    test -f /usr/lib/postgresql/18/lib/pg_hint_plan.so; \
    test -f /usr/share/postgresql/18/extension/pg_hint_plan.control; \
    cd /usr/lib/postgresql/18/lib; \
    sha256sum --check /usr/share/qorl/pg_hint_plan.so.sha256; \
    grep --fixed-strings "default_version = '${PG_HINT_PLAN_VERSION}'" \
        /usr/share/postgresql/18/extension/pg_hint_plan.control

# Run the full contract check during startup. The very long steady-state
# interval avoids health probes adding noise to timed benchmark queries; the
# orchestrator performs explicit pre/post checks for every run.
HEALTHCHECK --interval=168h --timeout=15s --start-period=1m --start-interval=1s --retries=1 \
    CMD ["/usr/local/bin/qorl-assert-benchmark-config"]

CMD ["postgres", "-c", "config_file=/etc/qorl/benchmark-v2.conf"]
