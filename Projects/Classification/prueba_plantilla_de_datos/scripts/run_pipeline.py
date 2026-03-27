"""
Pipeline completo: EDA → Features → Training → Predictions.

Ejecutar desde la raiz del proyecto:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --feature-set fs-002_cryo_interactions

Flags opcionales:
    --feature-set   Feature set a usar en 02_features y 03_train (default: fs-001_baseline)
    --skip-eda      Omite 01_eda.py
    --from-train    Ejecuta solo desde 03_train.py
    --predict-only  Ejecuta solo 04_predict.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

_ENV = {**os.environ, 'PYTHONUTF8': '1'}

# Scripts que aceptan --feature-set
_FS_SCRIPTS = {'02_features', '03_train'}

SCRIPTS = [
    ('01_eda',      'scripts/01_eda.py'),
    ('02_features', 'scripts/02_features.py'),
    ('03_train',    'scripts/03_train.py'),
    ('04_predict',  'scripts/04_predict.py'),
]


def run_script(name: str, path: str, extra_args: list | None = None) -> bool:
    """Ejecuta un script y retorna True si tuvo exito.

    Args:
        name: Nombre descriptivo del script para mostrar en consola.
        path: Ruta relativa al script desde la raiz del proyecto.
        extra_args: Argumentos adicionales a pasar al script.

    Returns:
        True si el proceso termino con codigo 0, False en caso contrario.
    """
    print(f'\n{"="*60}')
    print(f'  ▶  {name}')
    print(f'{"="*60}')
    cmd = [sys.executable, path] + (extra_args or [])
    result = subprocess.run(cmd, check=False, env=_ENV)
    if result.returncode != 0:
        print(f'\n❌ [ERROR] {name} fallo con codigo {result.returncode}')
        return False
    return True


def main() -> None:
    """Punto de entrada del pipeline orquestador."""
    parser = argparse.ArgumentParser(description='Pipeline ML Spaceship Titanic')
    parser.add_argument(
        '--feature-set',
        default=None,
        help='Feature set a usar en 02_features y 03_train. Ver src/features/feature_sets.py.',
    )
    parser.add_argument('--skip-eda',     action='store_true', help='Omite 01_eda.py')
    parser.add_argument('--from-train',   action='store_true', help='Ejecuta desde 03_train.py')
    parser.add_argument('--predict-only', action='store_true', help='Solo 04_predict.py')
    args = parser.parse_args()

    fs_args = ['--feature-set', args.feature_set] if args.feature_set else []

    scripts = SCRIPTS.copy()
    if args.predict_only:
        scripts = [s for s in scripts if s[0] == '04_predict']
    elif args.from_train:
        scripts = [s for s in scripts if s[0] in ('03_train', '04_predict')]
    elif args.skip_eda:
        scripts = [s for s in scripts if s[0] != '01_eda']

    for name, path in scripts:
        extra = fs_args if name in _FS_SCRIPTS else []
        if not run_script(name, path, extra_args=extra):
            sys.exit(1)

    print(f'\n{"="*60}')
    print('  ✅ Pipeline completado exitosamente')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
