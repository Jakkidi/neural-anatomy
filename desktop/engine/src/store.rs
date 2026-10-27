//! The weights of a loaded model, and the read-only questions the interface
//! asks about them: a rectangle of a matrix resampled for a plate, and the
//! statistics that set a plate's ink scale.

use std::collections::HashMap;
use std::sync::Mutex;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A rectangle of a stored tensor as the interface wants to see it.
/// With `transpose`, display rows are tensor columns.
#[derive(Debug, Clone)]
pub struct View {
    pub tensor: String,
    pub row0: usize,
    pub col0: usize,
    pub rows: usize,
    pub cols: usize,
    pub transpose: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

#[derive(Default)]
pub struct WeightStore {
    tensors: HashMap<String, Tensor>,
    stats: Mutex<HashMap<String, Stats>>,
}

impl WeightStore {
    pub fn new() -> WeightStore {
        WeightStore::default()
    }

    pub fn insert(&mut self, name: &str, shape: Vec<usize>, data: Vec<f32>) {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        self.tensors.insert(name.to_string(), Tensor { shape, data });
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("The weight \"{}\" is not loaded.", name))
    }

    pub fn data(&self, name: &str) -> Result<&[f32], String> {
        Ok(&self.get(name)?.data)
    }

    pub fn names(&self) -> Vec<(String, Vec<usize>)> {
        let mut out: Vec<_> = self.tensors.iter().map(|(name, tensor)| (name.clone(), tensor.shape.clone())).collect();
        out.sort();
        out
    }

    pub fn total(&self) -> usize {
        self.tensors.values().map(|tensor| tensor.data.len()).sum()
    }

    /// Rows and columns of a tensor seen as a matrix (a vector is one row).
    pub fn matrix_shape(&self, name: &str) -> Result<(usize, usize), String> {
        let shape = &self.get(name)?.shape;
        Ok(match shape.len() {
            0 => (1, 1),
            1 => (1, shape[0]),
            _ => (shape[0], shape[1..].iter().product()),
        })
    }

    /// The stored rectangle a view covers: (first row, first column, rows, columns).
    fn stored(&self, view: &View) -> Result<(usize, usize, usize, usize, usize), String> {
        let (r, c) = self.matrix_shape(&view.tensor)?;
        let (r0, c0, rows, cols) = if view.transpose { (view.col0, view.row0, view.cols, view.rows) } else { (view.row0, view.col0, view.rows, view.cols) };
        if rows == 0 || cols == 0 || r0 + rows > r || c0 + cols > c {
            return Err(format!("That region lies outside \"{}\".", view.tensor));
        }
        Ok((r0, c0, rows, cols, c))
    }

    /// A view resampled to `out_rows` x `out_cols`, row-major.
    ///
    /// When the output is smaller than the view, each cell holds the entry of
    /// largest magnitude in its block (sign kept), so a single large weight
    /// stays visible at any zoom. When it is the same size the values are exact.
    pub fn region(&self, view: &View, out_rows: usize, out_cols: usize) -> Result<Vec<f32>, String> {
        let (r0, c0, rows, cols, stride) = self.stored(view)?;
        if out_rows == 0 || out_cols == 0 || out_rows > view.rows || out_cols > view.cols {
            return Err("A region cannot be enlarged, only shrunk.".to_string());
        }
        let data = self.data(&view.tensor)?;
        let mut out = vec![0f32; out_rows * out_cols];
        // where each display row and column lands in the output
        let row_map: Vec<usize> = (0..view.rows).map(|r| (r * out_rows / view.rows).min(out_rows - 1)).collect();
        let col_map: Vec<usize> = (0..view.cols).map(|c| (c * out_cols / view.cols).min(out_cols - 1)).collect();
        // walk the tensor in memory order either way
        for r in 0..rows {
            let line = &data[(r0 + r) * stride + c0..(r0 + r) * stride + c0 + cols];
            if !view.transpose {
                let base = row_map[r] * out_cols;
                for (c, &v) in line.iter().enumerate() {
                    let cell = &mut out[base + col_map[c]];
                    if v.abs() > cell.abs() {
                        *cell = v;
                    }
                }
            } else {
                // stored row r is display column r; stored column c is display row c
                let column = col_map[r];
                for (c, &v) in line.iter().enumerate() {
                    let cell = &mut out[row_map[c] * out_cols + column];
                    if v.abs() > cell.abs() {
                        *cell = v;
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn stats(&self, view: &View) -> Result<Stats, String> {
        let key = format!("{}|{}|{}|{}|{}|{}", view.tensor, view.row0, view.col0, view.rows, view.cols, view.transpose);
        if let Some(found) = self.stats.lock().unwrap().get(&key) {
            return Ok(*found);
        }
        let (r0, c0, rows, cols, stride) = self.stored(view)?;
        let data = self.data(&view.tensor)?;
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut sum, mut squares) = (0f64, 0f64);
        for r in 0..rows {
            for &v in &data[(r0 + r) * stride + c0..(r0 + r) * stride + c0 + cols] {
                min = min.min(v);
                max = max.max(v);
                sum += v as f64;
                squares += (v as f64) * (v as f64);
            }
        }
        let count = rows * cols;
        let mean = sum / count as f64;
        let stats = Stats { min, max, mean, std: (squares / count as f64 - mean * mean).max(0.0).sqrt(), count };
        self.stats.lock().unwrap().insert(key, stats);
        Ok(stats)
    }
}
