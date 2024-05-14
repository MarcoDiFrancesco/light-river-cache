use crate::classification::mondrian_tree::MondrianTreeClassifier;
use ndarray::Array1;
use num::ToPrimitive;

use std::usize;

pub struct MondrianForestClassifier {
    trees: Vec<MondrianTreeClassifier>,
    n_labels: usize,
}
impl MondrianForestClassifier {
    pub fn new(n_trees: usize, n_features: usize, n_labels: usize) -> Self {
        let tree_default = MondrianTreeClassifier::new(n_features, n_labels);
        let trees = vec![tree_default; n_trees];
        MondrianForestClassifier { trees, n_labels }
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<f32>, y: usize) {
        for tree in &mut self.trees {
            tree.partial_fit(x, y);
        }
    }

    fn predict_proba(&self, x: &Array1<f32>) -> Array1<f32> {
        let mut tot_probs = Array1::zeros(self.n_labels);
        for tree in &self.trees {
            let probs = tree.predict_proba(x);
            assert!(
                !probs.iter().any(|&x| x.is_nan()),
                "Probability should not be NaN. Found: {:?}.",
                probs.to_vec()
            );
            tot_probs += &probs;
        }
        tot_probs /= self.trees.len().to_f32().unwrap();
        tot_probs
    }

    pub fn score(&mut self, x: &Array1<f32>, y: usize) -> f32 {
        let probs = self.predict_proba(x);
        let pred_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        if pred_idx == y {
            1.0
        } else {
            0.0
        }
    }

    pub fn cache_sort(&mut self) {
        for tree in &mut self.trees {
            tree.cache_sort();
        }
    }

    pub fn get_forest_size(&self) -> Vec<usize> {
        self.trees.iter().map(|t| t.get_tree_size()).collect()
    }
}
