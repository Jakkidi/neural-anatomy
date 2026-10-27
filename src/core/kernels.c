// Hot loops of the engine, compiled to WebAssembly SIMD.
// Build with tools/build-kernels.sh. The JS fallbacks in kernels.js compute
// the same thing, so the engine runs without this file.
//
// All pointers are byte offsets into the shared linear memory, which JS owns.

typedef unsigned int u32;

// y[t, :] = b + x[t, :] @ W          x: [n, nin]   W: [nin, nout]   y: [n, nout]
// (the layout of GPT-2's Conv1D and of a transposed nn.Linear)
__attribute__((export_name("affine")))
void affine(const float *restrict x, const float *restrict W, const float *restrict b,
            float *restrict y, u32 n, u32 nin, u32 nout) {
  for (u32 t = 0; t < n; t++) {
    float *yt = y + (unsigned long)t * nout;
    if (b) {
      for (u32 j = 0; j < nout; j++) yt[j] = b[j];
    } else {
      for (u32 j = 0; j < nout; j++) yt[j] = 0.0f;
    }
  }
  // One row of W stays in cache while it is applied to every position.
  for (u32 i = 0; i < nin; i++) {
    const float *wi = W + (unsigned long)i * nout;
    for (u32 t = 0; t < n; t++) {
      const float a = x[(unsigned long)t * nin + i];
      float *yt = y + (unsigned long)t * nout;
      for (u32 j = 0; j < nout; j++) yt[j] += a * wi[j];
    }
  }
}

// out[r] = dot(E[r, :], h)           E: [rows, n]   h: [n]
__attribute__((export_name("matvec")))
void matvec(const float *restrict E, const float *restrict h, float *restrict out, u32 rows, u32 n) {
  for (u32 r = 0; r < rows; r++) {
    const float *e = E + (unsigned long)r * n;
    float acc = 0.0f;
    for (u32 j = 0; j < n; j++) acc += e[j] * h[j];
    out[r] = acc;
  }
}

__attribute__((export_name("dot")))
float dot(const float *restrict a, const float *restrict b, u32 n) {
  float acc = 0.0f;
  for (u32 j = 0; j < n; j++) acc += a[j] * b[j];
  return acc;
}
