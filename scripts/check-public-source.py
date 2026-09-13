"""Validate the public checkout before CI installs or builds anything."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, check=True)
files = result.stdout.decode('utf-8').split('\0')
blocked_roots = ('data/', 'model/', 'config/', 'analysis/', 'research/', 'outputs/')
blocked_names = {'key.cmd', 'backend/supply_snapshot.json', 'backend/supply_snapshot.sha256', 'frontend/public/fleet-road-matrix.json'}
bad = [f for f in files if f and (f.startswith(blocked_roots) or f in blocked_names or
       (Path(f).name.startswith('.env') and Path(f).name != '.env.example') or
       Path(f).suffix.lower() in {'.parquet', '.csv', '.pem', '.key', '.p12', '.pfx', '.zip', '.xlsx', '.pptx', '.pdf', '.mp4'})]
if bad:
    print('Excluded publication artifacts are tracked:', *bad, sep='\n')
    sys.exit(1)
assert any(f.startswith('.kiro/') for f in files), 'Required Kiro specifications missing'
print('Public source boundary passed; .kiro specifications retained.')
