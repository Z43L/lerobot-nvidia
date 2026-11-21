# Pruebas offline del ACT (modeloACT/)

Solo inferencia y comparación con el dataset. No conecta con el robot ni envía comandos.

| Fichero | Para qué |
|---|---|
| `act_model.py` | Carga el checkpoint de `modeloACT/` en `mps` y predice un chunk de 100 acciones en grados. |
| `test_act.py` | Suite: carga y forward, error frente a acciones reales del dataset, rango de salidas. Escribe `results/latest.json`. |
| `diagnostico_act.py` | Comprueba la normalización (roundtrip) y la sensibilidad a imágenes y estado. |
| `act_cli.py` | Interfaz JSON para Claude: `predict --state=... --front a.png --gripper b.png` y `test`. |

```bash
./.venv312/bin/python pruebas_act/test_act.py --samples 60
./.venv312/bin/python pruebas_act/diagnostico_act.py
./.venv312/bin/python pruebas_act/act_cli.py test
./.venv312/bin/python pruebas_act/act_cli.py predict --state="-8.09,-15.38,22.51,79.82,166.68,14.15" --front pruebas_act/results/ejemplo_front.png --gripper pruebas_act/results/ejemplo_gripper.png
```

Notas:
- El estado va en grados; la pinza en 0..100. Usa `--state=` con valores negativos.
- Umbrales de aprobación definidos en `test_act.py` antes de ejecutar: error del primer paso < 5° y ≤ 2 % de acciones fuera de rango.
