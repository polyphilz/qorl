from pathlib import Path

from qorl.workload.query_structure import task_join_fingerprints
from qorl.workload.taskset import TaskSet
from scripts.benchmarks.audit_ceb_job_overlap import (
    build_report,
    job_fingerprints,
    read_ceb_sql,
)


def test_job_hashes_are_computed_from_task_structure(repository_root: Path) -> None:
    tasks = TaskSet.load(repository_root, "job").tasks
    graphs, topologies = job_fingerprints(tasks)
    graph, topology = task_join_fingerprints(tasks[0].model_dump())

    assert tasks[0].template_id in graphs[graph]
    assert tasks[0].template_id in topologies[topology]


def test_report_counts_tasks_and_templates_from_the_list(
    repository_root: Path, tmp_path: Path
) -> None:
    task_set = TaskSet.load(repository_root, "job")
    (tmp_path / "1a.sql").write_text(task_set.load_sql(task_set.tasks[0].model_dump()))
    report = build_report(
        read_ceb_sql(tmp_path, "sql"), task_set.tasks, "sql", None, None
    )

    assert report["job_inventory"] == {
        "inventory_id": "job",
        "task_count": 113,
        "template_count": 33,
    }
    assert report["summary"]["ceb_query_count"] == 1
    assert report["summary"]["excluded_template_count"] == 1
    assert "job-01" in report["templates"][0]["exact_job_template_matches"]
