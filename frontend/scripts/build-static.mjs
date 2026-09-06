import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

const executable = path.join(
  process.cwd(),
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'vinext.cmd' : 'vinext',
);
const result = spawnSync(executable, ['build'], {
  cwd: process.cwd(),
  encoding: 'utf8',
  shell: process.platform === 'win32',
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

const indexExists = existsSync(path.join(process.cwd(), 'dist', 'client', 'index.html'));
const combinedOutput = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
const knownWindowsShutdownAssertion =
  process.platform === 'win32' &&
  combinedOutput.includes('Build complete.') &&
  combinedOutput.includes('Assertion failed: !(handle->flags & UV_HANDLE_CLOSING)');

if (result.status === 0 && indexExists) process.exit(0);
if (knownWindowsShutdownAssertion && indexExists) {
  console.warn(
    '[build] Vinext completed the static export. Ignoring its known Windows-only shutdown assertion.',
  );
  process.exit(0);
}

if (!indexExists) console.error('[build] dist/client/index.html was not produced.');
process.exit(result.status || 1);
