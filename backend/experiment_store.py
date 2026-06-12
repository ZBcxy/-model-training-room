"""
实验管理：SQLite 持久化存储，训练历史 CRUD，配置快照，实验对比

所有训练实验的元数据和配置都存在这里。
"""

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================
# Configuration
# ============================================================

DB_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DB_DIR / "experiments.db"


# ============================================================
# Database Setup
# ============================================================

def get_connection() -> sqlite3.Connection:
    """获取数据库连接（自动创建表和目录）"""
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    """创建数据库表（如果不存在）"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            model_name TEXT DEFAULT '',
            dataset_name TEXT DEFAULT '',
            dataset_size INTEGER DEFAULT 0,
            finetuning_type TEXT DEFAULT 'lora',
            preset TEXT DEFAULT 'standard',
            status TEXT DEFAULT 'initialized',
            lora_rank INTEGER DEFAULT 16,
            lora_alpha INTEGER DEFAULT 32,
            learning_rate REAL DEFAULT 0.0002,
            num_epochs INTEGER DEFAULT 3,
            batch_size INTEGER DEFAULT 4,
            max_seq_length INTEGER DEFAULT 2048,
            chat_template TEXT DEFAULT '',
            final_loss REAL,
            best_loss REAL,
            training_duration_seconds REAL DEFAULT 0.0,
            vram_peak_gb REAL DEFAULT 0.0,
            config_json TEXT DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            tags TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            path TEXT NOT NULL,
            size_mb REAL DEFAULT 0.0,
            loss REAL,
            created_at REAL NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            epoch REAL DEFAULT 0.0,
            loss REAL NOT NULL,
            learning_rate REAL DEFAULT 0.0,
            grad_norm REAL DEFAULT 0.0,
            elapsed_seconds REAL DEFAULT 0.0,
            recorded_at REAL NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_metrics_exp ON training_metrics(experiment_id, step);
    """)


# ============================================================
# CRUD Operations - Experiments
# ============================================================

def create_experiment(
    experiment_id: str,
    model_id: str,
    model_name: str = "",
    dataset_name: str = "",
    dataset_size: int = 0,
    finetuning_type: str = "lora",
    preset: str = "standard",
    config: dict | None = None,
    tags: list[str] | None = None,
) -> dict:
    """
    创建新的实验记录。

    Returns:
        {"success": bool, "id": str, "error": str}
    """
    now = time.time()
    conn = get_connection()

    try:
        conn.execute("""
            INSERT INTO experiments (
                id, model_id, model_name, dataset_name, dataset_size,
                finetuning_type, preset,
                lora_rank, lora_alpha, learning_rate, num_epochs, batch_size,
                max_seq_length, chat_template,
                config_json, status, created_at, updated_at, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'initialized', ?, ?, ?)
        """, (
            experiment_id,
            model_id,
            model_name,
            dataset_name,
            dataset_size,
            finetuning_type,
            preset,
            config.get("lora_rank", 16) if config else 16,
            config.get("lora_alpha", 32) if config else 32,
            config.get("learning_rate", 2e-4) if config else 2e-4,
            config.get("num_epochs", 3) if config else 3,
            config.get("batch_size", 4) if config else 4,
            config.get("max_seq_length", 2048) if config else 2048,
            config.get("chat_template", "") if config else "",
            json.dumps(config or {}, ensure_ascii=False),
            now,
            now,
            json.dumps(tags or [], ensure_ascii=False),
        ))
        conn.commit()
        return {"success": True, "id": experiment_id, "error": ""}
    except sqlite3.IntegrityError:
        return {"success": False, "id": experiment_id, "error": "实验 ID 已存在"}
    except Exception as e:
        return {"success": False, "id": experiment_id, "error": str(e)}
    finally:
        conn.close()


def update_experiment_status(
    experiment_id: str,
    status: str,
    final_loss: float | None = None,
    best_loss: float | None = None,
    training_duration_seconds: float | None = None,
    vram_peak_gb: float | None = None,
) -> dict:
    """更新实验状态和结果"""
    now = time.time()
    conn = get_connection()

    updates = ["status = ?", "updated_at = ?"]
    params = [status, now]

    if final_loss is not None:
        updates.append("final_loss = ?")
        params.append(final_loss)
    if best_loss is not None:
        updates.append("best_loss = ?")
        params.append(best_loss)
    if training_duration_seconds is not None:
        updates.append("training_duration_seconds = ?")
        params.append(training_duration_seconds)
    if vram_peak_gb is not None:
        updates.append("vram_peak_gb = ?")
        params.append(vram_peak_gb)

    params.append(experiment_id)

    try:
        conn.execute(
            f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_experiment(experiment_id: str) -> dict | None:
    """获取单个实验的详情"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()

        if row is None:
            return None

        exp = dict(row)
        # Parse JSON fields
        exp["config"] = json.loads(exp.pop("config_json", "{}"))
        exp["tags"] = json.loads(exp.pop("tags", "[]"))

        # Get checkpoints
        checkpoints = conn.execute(
            "SELECT * FROM checkpoints WHERE experiment_id = ? ORDER BY step DESC",
            (experiment_id,)
        ).fetchall()
        exp["checkpoints"] = [dict(c) for c in checkpoints]

        return exp
    finally:
        conn.close()


def list_experiments(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    列出实验记录。

    Args:
        status: 按状态过滤（None 为全部）
        limit: 返回数量
        offset: 偏移量
    """
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                """SELECT id, model_id, model_name, dataset_name, finetuning_type,
                   status, final_loss, best_loss, training_duration_seconds,
                   created_at, updated_at
                   FROM experiments WHERE status = ?
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, model_id, model_name, dataset_name, finetuning_type,
                   status, final_loss, best_loss, training_duration_seconds,
                   created_at, updated_at
                   FROM experiments
                   ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_experiment(experiment_id: str) -> dict:
    """删除实验记录（级联删除 checkpoints 和 metrics）"""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


# ============================================================
# Training Metrics Logging
# ============================================================

def log_metric(
    experiment_id: str,
    step: int,
    loss: float,
    epoch: float = 0.0,
    learning_rate: float = 0.0,
    grad_norm: float = 0.0,
    elapsed_seconds: float = 0.0,
) -> dict:
    """记录一条训练指标"""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO training_metrics (experiment_id, step, epoch, loss, learning_rate, grad_norm, elapsed_seconds, recorded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, step, epoch, loss, learning_rate, grad_norm, elapsed_seconds, time.time()),
        )
        conn.commit()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def get_metrics(experiment_id: str, limit: int = 500) -> list[dict]:
    """获取实验的训练指标历史"""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT step, epoch, loss, learning_rate, grad_norm, elapsed_seconds
               FROM training_metrics WHERE experiment_id = ?
               ORDER BY step ASC LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ============================================================
# Comparison
# ============================================================

def compare_experiments(experiment_ids: list[str]) -> dict:
    """
    对比多个实验的关键指标。

    Returns:
        {
            "experiments": [dict per experiment],
            "best_loss_exp": str (id of experiment with lowest loss),
            "fastest_exp": str (id of fastest experiment),
        }
    """
    experiments = []
    for eid in experiment_ids:
        exp = get_experiment(eid)
        if exp:
            experiments.append({
                "id": exp["id"],
                "model_name": exp["model_name"],
                "dataset_name": exp["dataset_name"],
                "finetuning_type": exp["finetuning_type"],
                "final_loss": exp["final_loss"],
                "best_loss": exp["best_loss"],
                "training_duration_seconds": exp["training_duration_seconds"],
                "lora_rank": exp["lora_rank"],
                "learning_rate": exp["learning_rate"],
                "num_epochs": exp["num_epochs"],
                "status": exp["status"],
            })

    best_loss_exp = None
    best_loss = float("inf")
    fastest_exp = None
    fastest_time = float("inf")

    for exp in experiments:
        if exp["final_loss"] and exp["final_loss"] < best_loss:
            best_loss = exp["final_loss"]
            best_loss_exp = exp["id"]
        if exp["training_duration_seconds"] and exp["training_duration_seconds"] < fastest_time:
            fastest_time = exp["training_duration_seconds"]
            fastest_exp = exp["id"]

    return {
        "experiments": experiments,
        "best_loss_exp": best_loss_exp,
        "fastest_exp": fastest_exp,
    }


# ============================================================
# Statistics
# ============================================================

def get_statistics() -> dict:
    """
    获取全局统计数据。

    Returns:
        {
            "total_experiments": int,
            "completed_experiments": int,
            "failed_experiments": int,
            "total_training_hours": float,
            "avg_loss": float,
            "most_used_model": str,
            "most_used_dataset": str,
        }
    """
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) as c FROM experiments").fetchone()["c"]
        completed = conn.execute(
            "SELECT COUNT(*) as c FROM experiments WHERE status = 'completed'"
        ).fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) as c FROM experiments WHERE status = 'failed'"
        ).fetchone()["c"]
        total_hours = conn.execute(
            "SELECT COALESCE(SUM(training_duration_seconds), 0) as s FROM experiments"
        ).fetchone()["s"] / 3600.0
        avg_loss = conn.execute(
            "SELECT AVG(final_loss) as l FROM experiments WHERE final_loss IS NOT NULL"
        ).fetchone()["l"]

        most_model = conn.execute(
            "SELECT model_id, COUNT(*) as c FROM experiments GROUP BY model_id ORDER BY c DESC LIMIT 1"
        ).fetchone()
        most_dataset = conn.execute(
            "SELECT dataset_name, COUNT(*) as c FROM experiments GROUP BY dataset_name ORDER BY c DESC LIMIT 1"
        ).fetchone()

        return {
            "total_experiments": total,
            "completed_experiments": completed,
            "failed_experiments": failed,
            "total_training_hours": round(total_hours, 1),
            "avg_loss": round(avg_loss, 4) if avg_loss else 0.0,
            "most_used_model": most_model["model_id"] if most_model else "",
            "most_used_dataset": most_dataset["dataset_name"] if most_dataset else "",
        }
    finally:
        conn.close()


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("💾 实验管理数据库")
    print("=" * 60)
    print(f"  数据库路径: {DB_PATH}")

    # Clean test data from previous runs
    conn = get_connection()
    conn.execute("DELETE FROM experiments WHERE id LIKE 'test_%'")
    conn.commit()
    conn.close()

    # Test CRUD
    result = create_experiment(
        experiment_id="test_demo_001",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_name="Qwen2.5-7B",
        dataset_name="Chinese-Alpaca",
        dataset_size=50000,
        finetuning_type="lora",
        config={"lora_rank": 16, "learning_rate": 2e-4, "num_epochs": 3},
        tags=["中文", "对话"],
    )
    print(f"  创建实验: {result}")

    # Update
    update_experiment_status(
        "test_demo_001",
        status="completed",
        final_loss=1.24,
        best_loss=1.15,
        training_duration_seconds=11520,
    )
    exp = get_experiment("test_demo_001")
    print(f"  实验详情: id={exp['id']}, status={exp['status']}, loss={exp['final_loss']}")

    # List
    all_exp = list_experiments()
    print(f"  实验列表: {len(all_exp)} 条记录")

    # Stats
    stats = get_statistics()
    print(f"  统计: {stats}")

    # Cleanup
    delete_experiment("test_demo_001")
    print(f"  已清理测试数据")

    print()
    print("✅ 实验管理模块自检完成")
