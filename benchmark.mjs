// Four-stem SCNet ONNX inference benchmark: CPU or NVIDIA CUDA.
// Requires Node 22, FFmpeg, @audio/neural-separate and onnxruntime-node.
import separate from '@audio/neural-separate';
import { spawnSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import os from 'node:os';

const device = process.env.DEVICE || 'cpu';
// Native CPU is called 'node' by @audio/neural-runtime. CUDA is enabled in Dockerfile.gpu.
const backend = device === 'cpu' ? 'node' : device;
const model = process.env.MODEL || 'scnet';
const input = process.argv[2];
const rate = 44100;
const stemsToWrite = ['vocals', 'drums', 'bass', 'other'];

if (!input) {
  console.error('Usage: docker compose --profile cpu run --rm cpu /data/song.mp3');
  console.error('       docker compose --profile gpu run --rm gpu /data/song.mp3');
  process.exit(2);
}
if (!['cpu', 'cuda'].includes(device)) throw new Error(`Invalid DEVICE: ${device}`);
if (!['scnet', 'scnet-large'].includes(model)) throw new Error(`Invalid MODEL: ${model}`);

function run(command, args, options={}) {
  const result = spawnSync(command, args, { encoding: null, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} failed (exit ${result.status}): ${String(result.stderr || '').slice(-5000)}`);
  }
  return result.stdout;
}
function elapsed(t) { return Number(process.hrtime.bigint() - t) / 1e9; }
function round(n) { return Math.round(n * 100) / 100; }

async function main() {
  const name = basename(input, extname(input)).replace(/[^a-zA-Z0-9._-]/g, '_');
  const destination = join('/output', `${name}_${model}_${device}`);
  mkdirSync(destination, { recursive: true });
  console.log(`Host: ${os.cpus()[0]?.model || 'unknown'}; cores visible: ${os.cpus().length}`);
  console.log(`Model=${model}; device requested=${device}; backend=${backend}; input=${input}`);
  if (device === 'cuda') {
    const nvidia = run('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']);
    console.log(`NVIDIA GPU: ${nvidia.toString().trim()}`);
  }

  console.log('Decoding track to stereo float PCM at 44.1 kHz...');
  const decodeStart = process.hrtime.bigint();
  const pcm = run('ffmpeg', [
    '-hide_banner', '-nostdin', '-loglevel', 'error', '-i', input, '-map', 'a:0',
    '-vn', '-ac', '2', '-ar', String(rate), '-f', 'f32le', 'pipe:1',
  ], { maxBuffer: 1024 * 1024 * 512 });
  const frames = Math.floor(pcm.length / 8);
  if (frames === 0) throw new Error('No audio samples decoded.');
  const left = new Float32Array(frames), right = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    left[i] = pcm.readFloatLE(i * 8);
    right[i] = pcm.readFloatLE(i * 8 + 4);
  }
  const duration = frames / rate;
  const decodeSeconds = elapsed(decodeStart);
  console.log(`Decoded ${round(duration)} seconds in ${round(decodeSeconds)} seconds`);

  console.log('Separating (first run may also download/cache ONNX weights)...');
  const separationStart = process.hrtime.bigint();
  let lastProgress = -10;
  const result = await separate([left, right], {
    sampleRate: rate,
    model,
    device: backend,
    progress: ({chunk, totalChunks}) => {
      const percent = Math.round(100 * chunk / Math.max(1, totalChunks));
      if (percent >= lastProgress + 10 || percent >= 100) {
        console.log(`Progress: ${percent}% (${chunk}/${totalChunks})`);
        lastProgress = percent;
      }
    },
  });
  const separationSeconds = elapsed(separationStart);
  console.log(`Separation completed in ${round(separationSeconds)} seconds`);

  const outputRate = result.sampleRate || rate;
  const saveStart = process.hrtime.bigint();
  for (const stem of stemsToWrite) {
    const channels = result.stems[stem];
    if (!channels || channels.length !== 2) throw new Error(`Missing stereo stem: ${stem}`);
    const samples = Math.min(channels[0].length, channels[1].length);
    const interleaved = Buffer.allocUnsafe(samples * 8);
    for (let i = 0; i < samples; i++) {
      interleaved.writeFloatLE(channels[0][i], i * 8);
      interleaved.writeFloatLE(channels[1][i], i * 8 + 4);
    }
    run('ffmpeg', [
      '-hide_banner', '-nostdin', '-loglevel', 'error', '-y',
      '-f', 'f32le', '-ar', String(outputRate), '-ac', '2', '-i', 'pipe:0',
      '-c:a', 'pcm_f32le', join(destination, `${stem}.wav`),
    ], { input: interleaved, maxBuffer: 1024 * 1024 * 8 });
    console.log(`Saved ${stem}.wav`);
  }
  const saveSeconds = elapsed(saveStart);
  const stats = {
    input, model, device_requested: device, backend_requested: backend,
    audio_seconds: round(duration),
    decode_seconds: round(decodeSeconds),
    separation_seconds: round(separationSeconds),
    save_seconds: round(saveSeconds),
    separation_realtime_factor: round(separationSeconds / duration),
    separation_speed_x_realtime: round(duration / separationSeconds),
    process_rss_mb_at_end: round(process.memoryUsage().rss / 1024 ** 2),
    cpu: os.cpus()[0]?.model || 'unknown',
    cpu_threads_visible: os.cpus().length,
    output: destination,
    note: 'Includes model initialization and possibly initial weight download; repeat using cached weights.',
  };
  writeFileSync(join(destination, 'benchmark.json'), JSON.stringify(stats, null, 2) + '\n');
  console.log('\n=== BENCHMARK ===\n' + JSON.stringify(stats, null, 2));
  console.log(`\nOutput: ${destination}`);
}

main().catch(err => { console.error(err?.stack || err); process.exitCode = 1; });
