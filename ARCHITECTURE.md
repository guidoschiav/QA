# Dataset Auditor — Architecture

## Module Flow

```
Input (CSV/Parquet)
        │
        ▼
┌───────────────┐
│  config_loader│  ← configs/*.yaml
└──────┬────────┘
       │
       ▼
┌───────────────┐     ┌──────────────────┐
│  utils.py     │     │ snapshot_manager │ ← snapshots/*.parquet
│  detect format│     │ load prev snap   │
│  parse dates  │     └────────┬─────────┘
│  wide→long    │              │
└──────┬────────┘              │
       │                       │
       ▼                       ▼
┌──────────────────────────────────────┐
│              auditor.py              │
│         audit_dataset()              │
│                                      │
│  ┌──────────┐  ┌────────┐           │
│  │temporal  │  │schema  │           │
│  │ T1–T6    │  │ S1–S3  │           │
│  └──────────┘  └────────┘           │
│  ┌──────────┐  ┌──────────────────┐ │
│  │integrity │  │statistical       │ │
│  │ I1–I4    │  │ A1–A3            │ │
│  └──────────┘  └──────────────────┘ │
│  ┌────────────────────┐             │
│  │consistency C1–C3   │             │
│  └────────────────────┘             │
└──────────────────┬───────────────────┘
                   │
         List[CheckResult]
                   │
                   ▼
         ┌──────────────────┐
         │   report.py      │
         │  AuditReport     │
         │  print_human()   │
         │  export_json()   │
         └────────┬─────────┘
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  console output        reports/*.json
                             +
                    snapshots/*.parquet (updated)
```

## Complete Check List (19 checks)

### Temporalidad (T1–T6)
| ID | Name | Severity |
|----|------|----------|
| T1 | Frequency inference and validation | BLOCKER |
| T2 | Gap detection (missing dates by frequency) | BLOCKER |
| T3 | Duplicate records by Date + dimensions | BLOCKER |
| T4 | Staleness (last date vs today minus expected lag) | WARNING |
| T5 | Date range validation (start/end vs config) | WARNING |
| T6 | Day pattern detection (business days, weekends, holidays) | INFO |

### Schema / Metadata (S1–S3)
| ID | Name | Severity |
|----|------|----------|
| S1 | Required columns present | BLOCKER |
| S2 | Correct data types per column | WARNING |
| S3 | Dimension values vs expected list (new/disappeared entities) | WARNING |

### Integridad (I1–I4)
| ID | Name | Severity |
|----|------|----------|
| I1 | Null percentage per column | WARNING |
| I2 | Suspicious zeros in value column | WARNING |
| I3 | Exact duplicate rows | BLOCKER |
| I4 | Value range (min/max vs config) | WARNING |

### Estadística / Anomalías (A1–A3)
| ID | Name | Severity |
|----|------|----------|
| A1 | Z-score outliers per series/entity | WARNING |
| A2 | Abrupt percentage change vs rolling window | WARNING |
| A3 | Value distribution comparison vs snapshot | INFO |

### Consistencia inter-ejecución (C1–C3)
| ID | Name | Severity |
|----|------|----------|
| C1 | Historical rewrites detection (cells changed vs snapshot) | WARNING |
| C2 | Percentage of modified cells | INFO |
| C3 | Validation against revision whitelist | WARNING |

## Data Model

```python
@dataclass
class CheckResult:
    check_id: str           # e.g. "T1", "I2"
    check_name: str         # Human readable name
    category: str           # "temporal", "schema", "integrity", "statistical", "consistency"
    severity: str           # "BLOCKER", "WARNING", "INFO"
    passed: bool
    message: str            # Short summary
    details: dict           # Structured details (counts, lists, values)
    recommendation: str     # Actionable fix suggestion

@dataclass
class AuditReport:
    dataset_name: str
    timestamp: str          # ISO 8601
    total_checks: int
    passed: int
    failed: int
    blockers: int
    warnings: int
    infos: int
    checks: List[CheckResult]
    summary: str            # One-line overall verdict
```
