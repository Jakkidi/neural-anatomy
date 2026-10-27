//! The hot loops, shared by every architecture.
//!
//! They are written so the compiler vectorises them: SSE2 on any x86-64,
//! AVX2 where the processor has it (checked once at run time), NEON on Apple
//! silicon. Work is split across threads by position or by row.

use std::sync::OnceLock;

/// How many threads the kernels use.
pub fn threads() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        let wanted = std::env::var("NA_THREADS").ok().and_then(|text| text.parse::<usize>().ok());
        wanted.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)).clamp(1, 16)
    })
}

/// A name for the instruction set the kernels were compiled to use.
pub fn backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() { "AVX2" } else { "SSE2" }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "NEON"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| std::env::var_os("NA_NO_AVX2").is_none() && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

// ---- y[t, :] += x[t, i] * W[i, :] ---------------------------------------------------

/// One thread's share: rows `x` (n x nin) into rows `y` (n x nout), already
/// holding the bias. One row of W stays in cache while it is applied to every
/// position.
#[inline(always)]
fn affine_rows(x: &[f32], w: &[f32], y: &mut [f32], nin: usize, nout: usize) {
    let n = x.len() / nin;
    for i in 0..nin {
        let wi = &w[i * nout..(i + 1) * nout];
        for t in 0..n {
            let a = x[t * nin + i];
            let yt = &mut y[t * nout..(t + 1) * nout];
            for (out, weight) in yt.iter_mut().zip(wi) {
                *out += a * *weight;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_rows_avx2(x: &[f32], w: &[f32], y: &mut [f32], nin: usize, nout: usize) {
    affine_rows(x, w, y, nin, nout)
}

fn affine_chunk(x: &[f32], w: &[f32], y: &mut [f32], nin: usize, nout: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: the instruction sets were found on this processor
            return unsafe { affine_rows_avx2(x, w, y, nin, nout) };
        }
    }
    affine_rows(x, w, y, nin, nout)
}

/// `y = x @ W + b` for `x`: [n, nin], `W`: [nin, nout] stored row-major,
/// `y`: [n, nout]. This is the layout of GPT-2's Conv1D and of a transposed
/// `nn.Linear`.
pub fn affine(x: &[f32], w: &[f32], b: Option<&[f32]>, y: &mut [f32], n: usize, nin: usize, nout: usize) {
    assert!(x.len() >= n * nin && w.len() >= nin * nout && y.len() >= n * nout);
    let x = &x[..n * nin];
    let y = &mut y[..n * nout];
    for row in y.chunks_exact_mut(nout) {
        match b {
            Some(bias) => row.copy_from_slice(&bias[..nout]),
            None => row.fill(0.0),
        }
    }
    let all = threads();
    if all <= 1 || n * nin * nout < 400_000 {
        affine_chunk(x, w, y, nin, nout);
        return;
    }
    if n >= all {
        // many positions: each thread takes a share of them
        let rows = n.div_ceil(all);
        std::thread::scope(|scope| {
            for (xs, ys) in x.chunks(rows * nin).zip(y.chunks_mut(rows * nout)) {
                scope.spawn(move || affine_chunk(xs, w, ys, nin, nout));
            }
        });
        return;
    }
    // few positions (one, when a token is appended): each thread takes a share
    // of W's rows and sums into its own copy of y, and the copies are added up
    let share = nin.div_ceil(all);
    let parts: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..all)
            .filter(|k| k * share < nin)
            .map(|k| {
                let (from, to) = (k * share, ((k + 1) * share).min(nin));
                scope.spawn(move || {
                    let mut part = vec![0f32; n * nout];
                    // the same loop as affine_rows, over a band of W and of x's columns
                    let band: Vec<f32> = (0..n).flat_map(|t| x[t * nin + from..t * nin + to].iter().copied()).collect();
                    affine_chunk(&band, &w[from * nout..to * nout], &mut part, to - from, nout);
                    part
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().expect("a kernel thread panicked")).collect()
    });
    for part in &parts {
        for (out, value) in y.iter_mut().zip(part) {
            *out += *value;
        }
    }
}

// ---- dot products ---------------------------------------------------------------------

/// Eight running sums, so the additions can go eight wide.
#[inline(always)]
fn dot_body(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let mut pa = a.chunks_exact(8);
    let mut pb = b.chunks_exact(8);
    for (ca, cb) in (&mut pa).zip(&mut pb) {
        for k in 0..8 {
            lanes[k] += ca[k] * cb[k];
        }
    }
    let mut sum = 0f32;
    for (x, y) in pa.remainder().iter().zip(pb.remainder()) {
        sum += x * y;
    }
    sum + ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matvec_rows_avx2(e: &[f32], h: &[f32], out: &mut [f32]) {
    matvec_rows(e, h, out)
}

#[inline(always)]
fn matvec_rows(e: &[f32], h: &[f32], out: &mut [f32]) {
    let n = h.len();
    for (value, row) in out.iter_mut().zip(e.chunks_exact(n)) {
        *value = dot_body(row, h);
    }
}

fn matvec_chunk(e: &[f32], h: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: the instruction sets were found on this processor
            return unsafe { matvec_rows_avx2(e, h, out) };
        }
    }
    matvec_rows(e, h, out)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_body(a, b)
}

/// `out[r] = E[r, :] . h` for `E`: [rows, n]. With `parallel`, rows are split
/// across threads; a caller that is already one of several threads passes false.
pub fn matvec(e: &[f32], h: &[f32], out: &mut [f32], parallel: bool) {
    let n = h.len();
    let rows = out.len();
    assert!(e.len() >= rows * n);
    let e = &e[..rows * n];
    let workers = if parallel { threads().min(rows / 2048 + 1) } else { 1 };
    if workers <= 1 {
        matvec_chunk(e, h, out);
        return;
    }
    let share = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (es, os) in e.chunks(share * n).zip(out.chunks_mut(share)) {
            scope.spawn(move || matvec_chunk(es, h, os));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_matches_the_definition() {
        let (n, nin, nout) = (37, 19, 53);
        let mut seed = 7u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let x: Vec<f32> = (0..n * nin).map(|_| next()).collect();
        let w: Vec<f32> = (0..nin * nout).map(|_| next()).collect();
        let b: Vec<f32> = (0..nout).map(|_| next()).collect();
        let mut y = vec![9.0f32; n * nout];
        affine(&x, &w, Some(&b), &mut y, n, nin, nout);
        for t in 0..n {
            for j in 0..nout {
                let want: f64 = b[j] as f64 + (0..nin).map(|i| x[t * nin + i] as f64 * w[i * nout + j] as f64).sum::<f64>();
                assert!((y[t * nout + j] as f64 - want).abs() < 1e-4);
            }
        }
        // one position, wide enough to take the split-by-rows path
        let (nin2, nout2) = (700, 640);
        let x2: Vec<f32> = (0..nin2).map(|_| next()).collect();
        let w2: Vec<f32> = (0..nin2 * nout2).map(|_| next()).collect();
        let b2: Vec<f32> = (0..nout2).map(|_| next()).collect();
        let mut y2 = vec![0f32; nout2];
        affine(&x2, &w2, Some(&b2), &mut y2, 1, nin2, nout2);
        for j in 0..nout2 {
            let want: f64 = b2[j] as f64 + (0..nin2).map(|i| x2[i] as f64 * w2[i * nout2 + j] as f64).sum::<f64>();
            assert!((y2[j] as f64 - want).abs() < 1e-3, "column {j}: {} vs {want}", y2[j]);
        }
        let mut out = vec![0f32; nin];
        matvec(&w, &x[..nout], &mut out, true);
        for i in 0..nin {
            let want: f64 = (0..nout).map(|j| w[i * nout + j] as f64 * x[j] as f64).sum();
            assert!((out[i] as f64 - want).abs() < 1e-4);
        }
    }
}
