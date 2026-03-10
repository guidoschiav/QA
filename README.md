# Dataset Auditor

Sistema automático de auditoría de calidad de datos para datasets publicados por scrapers/conectores.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Auditar un CSV
python run_audit.py --data mi_dataset.csv --config configs/example_config.yaml

# Auditar un parquet
python run_audit.py --data mi_dataset.parquet --config configs/my_config.yaml

# Guardar reporte JSON
python run_audit.py --data mi_dataset.csv --config configs/example_config.yaml --report-dir reports/
```

## Configuración

Copiar `configs/example_config.yaml` y ajustar los campos. Cada dataset tiene su propio YAML.

Campos clave:
- `format`: `"long"` o `"wide"`
- `expected_frequency`: `"B"` (business days), `"D"` (daily), `"W"` (weekly), `"M"` (monthly)
- `expected_dimensions`: mapa de columna → lista de valores válidos (null = no validar)
- `revision_whitelist`: configuración de revisiones históricas esperadas

## Tests

```bash
python -m pytest tests/ -v
```

## Estructura de Checks

| Categoría | Checks | Descripción |
|-----------|--------|-------------|
| Temporal | T1–T6 | Frecuencia, gaps, duplicados, staleness, rango, patrones |
| Schema | S1–S3 | Columnas, tipos, dimensiones |
| Integridad | I1–I4 | Nulos, ceros, duplicados exactos, rangos |
| Estadística | A1–A3 | Z-score, cambios abruptos, distribución |
| Consistencia | C1–C3 | Reescrituras, % modificado, whitelist |

## Severidades

- `BLOCKER`: Error crítico, el dataset no debería publicarse
- `WARNING`: Problema que requiere revisión
- `INFO`: Información de monitoreo, sin acción urgente
