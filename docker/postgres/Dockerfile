# Pin the multi-platform index. compose.yaml deliberately selects linux/amd64,
# which resolves to the amd64 manifest recorded in versions.json.
ARG POSTGRES_BASE_IMAGE=postgres:16.15-bookworm@sha256:bb3e1a57e5407e0a5280b4211980a5e537f4abd234a87014ac979849a78dd825

FROM ${POSTGRES_BASE_IMAGE} AS pg_hint_plan_builder

ARG PG_HINT_PLAN_TAG=REL16_1_6_2
ARG PG_HINT_PLAN_SOURCE_SHA256=657ff36bbfa8f9659af9faea48e497055b0e72715fb327eb72ab9b967200dc4e
ARG POSTGRES_DEV_PACKAGE_VERSION=16.15-1.pgdg12+2

RUN set -eux; \
    apt-get update; \
    apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        "postgresql-server-dev-16=${POSTGRES_DEV_PACKAGE_VERSION}"; \
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
    mkdir -p /opt/pg_hint_plan-install/usr/share/qprl; \
    dpkg-query --show --showformat='${binary:Package}\t${Version}\n' \
        | LC_ALL=C sort \
        > /opt/pg_hint_plan-install/usr/share/qprl/extension-build-packages.tsv; \
    cd /opt/pg_hint_plan-install/usr/lib/postgresql/16/lib; \
    sha256sum pg_hint_plan.so \
        > /opt/pg_hint_plan-install/usr/share/qprl/pg_hint_plan.so.sha256

FROM ${POSTGRES_BASE_IMAGE}

ARG PG_HINT_PLAN_TAG=REL16_1_6_2
ARG PG_HINT_PLAN_VERSION=1.6.2
ARG PG_HINT_PLAN_COMMIT=b35a03c789b5707a4f96e2ee719101d4f7016a88
ARG PG_HINT_PLAN_SOURCE_SHA256=657ff36bbfa8f9659af9faea48e497055b0e72715fb327eb72ab9b967200dc4e

LABEL org.opencontainers.image.title="qprl-postgres" \
      org.opencontainers.image.description="Pinned PostgreSQL and pg_hint_plan runtime for QPRL plan-steering experiments" \
      org.opencontainers.image.version="16.15-pg_hint_plan-1.6.2" \
      org.opencontainers.image.source="https://github.com/ossc-db/pg_hint_plan" \
      io.qprl.postgresql.version="16.15" \
      io.qprl.pg_hint_plan.version="${PG_HINT_PLAN_VERSION}" \
      io.qprl.pg_hint_plan.tag="${PG_HINT_PLAN_TAG}" \
      io.qprl.pg_hint_plan.commit="${PG_HINT_PLAN_COMMIT}" \
      io.qprl.pg_hint_plan.source-sha256="${PG_HINT_PLAN_SOURCE_SHA256}"

ENV PG_HINT_PLAN_VERSION=${PG_HINT_PLAN_VERSION} \
    PG_HINT_PLAN_TAG=${PG_HINT_PLAN_TAG} \
    PG_HINT_PLAN_COMMIT=${PG_HINT_PLAN_COMMIT}

COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/lib/postgresql/16/lib/ /usr/lib/postgresql/16/lib/
COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/share/postgresql/16/extension/ /usr/share/postgresql/16/extension/
COPY --from=pg_hint_plan_builder /opt/pg_hint_plan-install/usr/share/qprl/ /usr/share/qprl/

COPY postgresql.conf /etc/qprl/postgresql.conf
COPY --chmod=0555 initdb/001-qprl-bootstrap.sh /docker-entrypoint-initdb.d/001-qprl-bootstrap.sh

RUN set -eux; \
    test -f /usr/lib/postgresql/16/lib/pg_hint_plan.so; \
    test -f /usr/share/postgresql/16/extension/pg_hint_plan.control; \
    cd /usr/lib/postgresql/16/lib; \
    sha256sum --check /usr/share/qprl/pg_hint_plan.so.sha256; \
    grep --fixed-strings "default_version = '${PG_HINT_PLAN_VERSION}'" \
        /usr/share/postgresql/16/extension/pg_hint_plan.control

HEALTHCHECK --interval=5s --timeout=5s --start-period=10s --retries=12 \
    CMD pg_isready --username="$POSTGRES_USER" --dbname="${POSTGRES_DB:-$POSTGRES_USER}" || exit 1

CMD ["postgres", "-c", "config_file=/etc/qprl/postgresql.conf"]

