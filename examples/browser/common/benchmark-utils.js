function getSupportedOps() {
  return Array.from(
    document.querySelectorAll('input[name=supportedOp]:checked')).map(x => x.value);
}

function printHeader(msg) {
  predictions.innerHTML += `<p><b>${msg}</b></p>`;
}

function log(msg = '') {
  console.log(msg);
  predictions.innerHTML += `${msg}<br>`;
}

async function runBenchmark(session, inputData, iterations, profiling = false) {
  log();
  log('Running benchmark (Check progress in the console):');
  if (profiling) {
    session.startProfiling();
  }
  const timingResults = [];
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await session.run([inputData]);
    timingResults.push(performance.now() - start);
    console.log(`${i + 1}/${iterations}`);
  }
  const {mean, std} = statUtils(timingResults);
  log(`${(mean).toFixed(3)}${isNaN(std) ? '' : ` ± ${std.toFixed(3)}`} ms`);
  if (profiling) {
    session.endProfiling();
  }
}

function statUtils(arr) {
  const d = arr.reduce((d, v) => {
    d.sum += v;
    d.sum2 += v * v;
    return d;
  }, {
    sum: 0,
    sum2: 0
  });
  const len = arr.length;
  const mean = d.sum / len;
  const std  = Math.sqrt((d.sum2 - len * mean * mean) / (len - 1));
  return {
    mean: mean,
    std: std
  };
}
