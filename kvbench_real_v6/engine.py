"""Local process/telemetry support, adapted from the supplied v4 controller."""
from __future__ import annotations
import csv, gzip, hashlib, json, os, shlex, signal, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path
from typing import Any
from metrics import atomic_json, finite, parse_prometheus
HERE = Path(__file__).resolve().parent


def log(message: str) -> None:
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'), message, flush=True)

def run_text(cmd: list[str], timeout: int = 90, env: dict[str, str] | None = None) -> str:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout, env=env)
    if p.returncode:
        raise RuntimeError(f'{shlex.join(cmd)}\n{p.stderr[-6000:]}')
    return p.stdout.strip()

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0

def http_text(url: str, timeout: int = 5) -> str:
    # Do not send localhost benchmark traffic through HTTP_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read().decode('utf-8')

def conda_command(env_name: str, tail: list[str]) -> list[str]:
    return ['bash', '-c', 'set -eo pipefail; source "$1"; conda activate "$2"; '
            'unset PYTHONHOME PYTHONPATH; shift 2; exec "$@"',
            'kvbench-child', os.environ['CONDA_SH'], env_name, *tail]

def process_snapshot() -> dict[int, tuple[int, str, str]]:
    out = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            out[int(path.parent.name)] = (int(fields[1]), fields[19], fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return out

def owned_descendants(root: int) -> dict[int, str]:
    snap = process_snapshot(); owned = {root}
    while True:
        next_set = owned | {pid for pid, (ppid, _, _) in snap.items() if ppid in owned}
        if next_set == owned:
            break
        owned = next_set
    return {pid: snap[pid][1] for pid in owned if pid in snap}

def stop_process_group(proc: subprocess.Popen | None, grace: float = 15) -> None:
    if proc is None:
        return
    owned = owned_descendants(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Also include descendants which deliberately created a new session.
    for pid, stamp in owned.items():
        snap = process_snapshot().get(pid)
        if snap and snap[1] == stamp and snap[2] != 'Z':
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()
        snap = process_snapshot()
        alive = [pid for pid, stamp in owned.items()
                 if pid in snap and snap[pid][1] == stamp and snap[pid][2] != 'Z']
        if not alive:
            break
        time.sleep(0.2)
    snap = process_snapshot()
    for pid, stamp in owned.items():
        if pid in snap and snap[pid][1] == stamp and snap[pid][2] != 'Z':
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

def gpu_processes(gpu: str) -> str:
    return run_text(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid,process_name',
                     '--format=csv,noheader,nounits'], timeout=10)

def ensure_gpu_idle(gpu: str) -> None:
    active = gpu_processes(gpu)
    if active and 'No running processes found' not in active:
        raise RuntimeError(f'GPU {gpu} 已有计算进程，控制脚本不会终止它们：\n{active}\n请先手动停止旧冒烟服务。')

class Telemetry:
    def __init__(self, path: Path, base: str, gpu: str, interval: float):
        self.path, self.base, self.gpu, self.interval = path, base, gpu, interval
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.done.set(); self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('采样线程未正常退出；停止实验以避免测量窗口重叠。')

    def loop(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        metrics_dir = self.path / 'metrics.samples'; metrics_dir.mkdir(exist_ok=True)
        columns = ['memory_used_mib', 'gpu_util_pct', 'temperature_c', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz']
        query = 'memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem'
        counter = 0
        with (self.path / 'telemetry.jsonl').open('a', encoding='utf-8') as target:
            while not self.done.is_set():
                started = time.monotonic()
                row: dict[str, Any] = {'unix_time': time.time(), 'monotonic_time': started}
                try:
                    text = http_text(self.base + '/metrics', 3)
                    with gzip.open(metrics_dir / f'{counter:06d}.prom.gz', 'wt', encoding='utf-8') as f:
                        f.write(text)
                    metrics = parse_prometheus(text); gauges: dict[str, float] = {}
                    for k, value in metrics.items():
                        name = k.split('{', 1)[0]
                        if name in ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc',
                                    'vllm:num_requests_waiting', 'vllm:num_requests_running'):
                            gauges[name] = gauges.get(name, 0) + value
                    row['gauges'] = gauges
                except Exception as exc:
                    row['metrics_error'] = str(exc)
                try:
                    text = run_text(['nvidia-smi', '-i', self.gpu, '--query-gpu=' + query,
                                     '--format=csv,noheader,nounits'], timeout=4)
                    values = next(csv.reader([text.splitlines()[0]]))
                    row['gpu'] = {k: finite(v.strip()) for k, v in zip(columns, values)}
                except Exception as exc:
                    row['gpu_error'] = str(exc)
                target.write(json.dumps(row, ensure_ascii=False) + '\n'); target.flush()
                counter += 1
                self.done.wait(max(0.1, self.interval - (time.monotonic() - started)))

def identity(cfg: dict[str, Any]) -> dict[str, Any]:
    kit = Path(cfg['kit']); model = Path(cfg['model'])
    files = {str(p): sha(p) for p in sorted(HERE.glob('*.py'))}
    files.update({str(p): sha(p) for p in sorted(HERE.glob('*.sh'))})
    files.update({str(p): sha(p) for p in (kit / 'serve.sh', kit / 'env.sh')})
    tokfiles = [p for p in sorted(model.iterdir()) if p.is_file() and
                (p.suffix in ('.json', '.jinja', '.txt', '.model') and p.stat().st_size < 100_000_000)]
    info = {
        'scripts_sha256': files, 'model_config_tokenizer_sha256': {p.name: sha(p) for p in tokfiles},
        'weight_file_sizes_mtime_ns': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                                     for p in sorted(model.glob('*.safetensors'))},
        'python': sys.version, 'eval_packages': run_text([sys.executable, '-m', 'pip', 'freeze']),
        'serve_packages': run_text(conda_command(cfg['serve_env'], ['python', '-m', 'pip', 'freeze'])),
        'gpu': run_text(['nvidia-smi', '-i', cfg['gpu'], '--query-gpu=uuid,name,driver_version,memory.total,power.limit',
                         '--format=csv,noheader,nounits'], timeout=10),
    }
    code = ('import json,pathlib,subprocess,vllm,torch; '
            'r=pathlib.Path(vllm.__file__).resolve().parents[1]; '
            'g=lambda *a: subprocess.run(["git","-C",str(r),*a],capture_output=True,text=True).stdout; '
            'print("KVBENCH_IDENTITY="+json.dumps({"module":str(vllm.__file__),"torch":torch.__version__, '
            '"cuda":torch.version.cuda,"commit":g("rev-parse","HEAD"), '
            '"status":g("status","--porcelain"),"diff":g("diff","HEAD")}))')
    raw = run_text(conda_command(cfg['serve_env'], ['python', '-c', code]))
    lines = [line.split('KVBENCH_IDENTITY=', 1)[1] for line in raw.splitlines() if line.startswith('KVBENCH_IDENTITY=')]
    if len(lines) != 1:
        raise ValueError('服务环境标识未成功导出，检查 Python/vLLM 导入日志。')
    info['source'] = json.loads(lines[0])
    return info

class ServerManager:

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.root = Path(cfg['run_dir'])
        self.base = f"http://127.0.0.1:{cfg['port']}"
        self.server: subprocess.Popen | None = None
        self.server_file = None
        self.server_group = None
        self.server_folder: Path | None = None
        self.tokenizer = None
        self.body_cache: dict[int, str] = {}
        self._counter = 0

    def stop(self) -> None:
        stop_process_group(self.server)
        self.server = None
        self.server_group = None
        if self.server_file:
            self.server_file.close()
            self.server_file = None

    def start(self, p: dict[str, Any]) -> None:
        group = p['id'] if self.cfg['preset'] == 'prefix' else (p['repeat'], p['method'], p['apc'])
        if self.server is not None and self.server.poll() is None and (group == self.server_group):
            return
        self.stop()
        if port_busy(self.cfg['port']):
            raise RuntimeError('端口仍被占用；为避免误测其他服务，停止启动。请检查残留进程。')
        ensure_gpu_idle(self.cfg['gpu'])
        self._counter += 1
        self.server_folder = self.root / 'servers' / f"{time.time_ns()}_{self._counter}_{p['method']}"
        self.server_folder.mkdir(parents=True)
        self.server_file = (self.server_folder / 'launcher.log').open('w')
        env = os.environ.copy()
        env.update(KVBENCH_ROOT=self.cfg['root'], OUT=str(self.server_folder), MODEL=self.cfg['model'], PORT=str(self.cfg['port']), CUDA_VISIBLE_DEVICES=self.cfg['gpu'], APC=str(p['apc']), MAXLEN=str(self.cfg['max_len']), MAXSEQ=str(self.cfg['max_seqs']), BATCHED_TOKENS=str(self.cfg['batch_tokens']), GPU_UTIL=str(self.cfg['gpu_util']), KV_BYTES=str(self.cfg.get('kv_bytes') or ''), PYTHONNOUSERSITE='1', KVBENCH_SERVE_ENV=self.cfg['serve_env'])
        env.pop('PYTHONHOME', None)
        env.pop('PYTHONPATH', None)
        args = ['bash', str(Path(self.cfg['kit']) / 'serve.sh'), p['method']]
        if self.cfg['eager']:
            args.append('--enforce-eager')
        command = conda_command(self.cfg['serve_env'], args)
        atomic_json(self.server_folder / 'launch.json', {'command': command, 'point': p})
        log(f"启动 {p['method']} APC={p['apc']} eager={self.cfg['eager']}；日志：{self.server_folder / 'launcher.log'}")
        self.server = subprocess.Popen(command, env=env, stdout=self.server_file, stderr=subprocess.STDOUT, cwd=self.cfg['kit'], start_new_session=True)
        deadline = time.monotonic() + self.cfg['startup_timeout']
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(f"服务启动退出，请检查 {self.server_folder / 'launcher.log'}")
            try:
                http_text(self.base + '/health', 2)
                data = json.loads(http_text(self.base + '/v1/models', 2))
                if 'qwen-kvbench' not in [m['id'] for m in data.get('data', [])]:
                    raise RuntimeError('监听的服务没有预期模型名 qwen-kvbench。')
                self.server_group = group
                return
            except (OSError, ValueError):
                time.sleep(2)
        raise TimeoutError(f"启动超时 {self.cfg['startup_timeout']} 秒。")

    def invoke(self, cmd: list[str], logfile: Path) -> None:
        logfile.with_suffix('.command.txt').write_text(shlex.join(cmd) + '\n')
        env = os.environ.copy()
        env['NO_PROXY'] = '127.0.0.1,localhost' + (',' + env['NO_PROXY'] if env.get('NO_PROXY') else '')
        env['no_proxy'] = env['NO_PROXY']
        env['PYTHONNOUSERSITE'] = '1'
        proc = None
        try:
            with logfile.open('w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                deadline = time.monotonic() + self.cfg['point_timeout']
                while proc.poll() is None:
                    if self.server is None or self.server.poll() is not None:
                        raise RuntimeError('测量期间服务端退出。')
                    if time.monotonic() > deadline:
                        raise TimeoutError('此阶段超过 point-timeout；配置未被自动降低。')
                    time.sleep(0.5)
                if proc.returncode:
                    raise RuntimeError(f'EvalScope 返回 {proc.returncode}，日志：{logfile}')
        finally:
            if proc is not None and proc.poll() is None:
                stop_process_group(proc, grace=3)
