use ndarray::{Array1, Array2};
use num::ToPrimitive;
use std::fmt;
use std::usize;

/// Node struct
#[derive(Clone)]
pub struct Node {
    pub parent: Option<usize>,
    pub time: f32, // Time: how much I increased the size of the box
    pub is_leaf: bool,
    pub min_list: Array1<f32>, // Lists representing the minimum and maximum values of the data points contained in the current node
    pub max_list: Array1<f32>,
    pub feature: usize, // Feature in which a split occurs
    pub threshold: f32, // Threshold in which the split occures
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub stats: Stats,
}
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node<left={:?}, right={:?}, parent={:?}, time={:.3}, min={:?}, max={:?}, counts={:?}>",
            self.left,
            self.right,
            self.parent,
            self.time,
            self.min_list.to_vec(),
            self.max_list.to_vec(),
            self.stats.counts.to_vec(),
        )?;
        Ok(())
    }
}

impl Node {
    pub fn add_to_leaf(&mut self, x: &Array1<f32>, y: usize) {
        self.stats.add(x, y);
    }
    pub fn get_stats_from_children(&self, left_s: &Stats, right_s: &Stats) -> Stats {
        left_s.merge(right_s)
    }
    /// Check if all the labels are the same in the node.
    /// e.g. y=2, stats.counts=[0, 1, 10] -> False
    /// e.g. y=2, stats.counts=[0, 0, 10] -> True
    /// e.g. y=1, stats.counts=[0, 0, 10] -> False
    pub fn is_dirac(&self, y: usize) -> bool {
        self.stats.counts.sum() == self.stats.counts[y]
    }
}

/// Stats assocociated to one node
///
/// In nel215 code it is "Classifier"
#[derive(Clone)]
pub struct Stats {
    sums: Array2<f32>,
    sq_sums: Array2<f32>,
    pub counts: Array1<usize>,
    n_labels: usize,
}
impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n┌ Stats")?;
        // sums
        write!(f, "│ sums: [")?;
        for row in self.sums.outer_iter() {
            write!(f, "{:?}, ", row.to_vec())?;
        }
        writeln!(f, "]")?;
        // sq_sums
        write!(f, "│ sq_sums: [")?;
        for row in self.sq_sums.outer_iter() {
            write!(f, "{:?}, ", row.to_vec())?;
        }
        writeln!(f, "]")?;
        // count
        write!(f, "└ counts: {}", self.counts)?;
        Ok(())
    }
}
impl Stats {
    pub fn new(n_labels: usize, n_features: usize) -> Self {
        Stats {
            sums: Array2::<f32>::zeros((n_labels, n_features)),
            sq_sums: Array2::<f32>::zeros((n_labels, n_features)),
            counts: Array1::zeros(n_labels),
            n_labels,
        }
    }
    pub fn create_result(&self, x: &Array1<f32>, w: f32) -> Array1<f32> {
        let probs = self.predict_proba(x);
        probs * w
    }
    fn add(&mut self, x: &Array1<f32>, y: usize) {
        // Same as: self.sums[y] += x;
        self.sums.row_mut(y).zip_mut_with(&x, |a, &b| *a += b);

        // Same as: self.sq_sums[y] += x*x;
        // e.g. x: [1.059 0.580] -> x*x: [1.122  0.337]
        self.sq_sums
            .row_mut(y)
            .zip_mut_with(&x, |a, &b| *a += b * b);
        self.counts[y] += 1;
    }
    fn merge(&self, s: &Stats) -> Stats {
        Stats {
            sums: self.sums.clone() + &s.sums,
            sq_sums: self.sq_sums.clone() + &s.sq_sums,
            counts: self.counts.clone() + &s.counts,
            n_labels: self.n_labels,
        }
    }
    pub fn predict_proba(&self, x: &Array1<f32>) -> Array1<f32> {
        let mut probs = Array1::zeros(self.n_labels);
        let mut sum_prob = 0.0;

        // println!("predict_proba() - start {}", self);

        for (index, ((sum, sq_sum), &count)) in self
            .sums
            .outer_iter()
            .zip(self.sq_sums.outer_iter())
            .zip(self.counts.iter())
            .enumerate()
        {
            let epsilon = f32::EPSILON;
            let count_f = count.to_f32().unwrap();
            let avg = &sum / count_f;
            let var = (&sq_sum / count_f) - (&avg * &avg) + epsilon;
            let sigma = (&var * count_f) / (count_f - 1.0 + epsilon);
            let pi = std::f32::consts::PI * 2.0;
            let z = pi.powi(x.len() as i32) * sigma.mapv(|s| s * s).sum().sqrt();
            // Dot product
            let dot_feature = (&(x - &avg) * &(x - &avg)).sum();
            let dot_sigma = (&sigma * &sigma).sum();
            let exponent = -0.5 * dot_feature / dot_sigma;
            // epsilon added since exponent.exp() could be zero if exponent is very small
            let mut prob = (exponent.exp() + epsilon) / z;
            if count <= 0 {
                assert!(prob.is_nan(), "Probabaility should be NaN. Found: {prob}.");
                prob = 0.0;
            }
            sum_prob += prob;
            probs[index] = prob;
        }

        // Check at least one probability is non-zero. Otherwise we have division by zero.
        assert!(
            !probs.iter().all(|&x| x == 0.0),
            "At least one probability should not be zero. Found: {:?}.",
            probs.to_vec()
        );

        for prob in probs.iter_mut() {
            *prob /= sum_prob;
        }
        // println!("predict_proba() post - probs: {:?}", probs.to_vec());
        probs
    }
}
