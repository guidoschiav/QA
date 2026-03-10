# Dataset Auditor

Sistema automático de auditoría de calidad de datos.

## Stack
- Python 3.10+, pandas, numpy, pyyaml, scipy
- Snapshots en parquet, configs en YAML, reportes en JSON

## Convenciones
- Funciones de checks retornan List[CheckResult] (dataclass)
- Severidades: BLOCKER, WARNING, INFO
- Config por dataset en configs/*.yaml
- Los checks deben ser vectorizados (no iterrows)
- Formato de fecha siempre YYYY-MM-DD
- Soporta datasets long (Date+dims+Value) y wide (Date+múltiples columnas numéricas)

## Comandos
- Correr auditoría: python run_audit.py --data archivo.csv --config configs/example_config.yaml
- Tests: python -m pytest tests/ -v
