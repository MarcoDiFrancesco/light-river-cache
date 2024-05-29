use crate::classification::alias::FType;
use crate::classification::mondrian_tree::MondrianTreeClassifier;

use ndarray::Array1;

use std::usize;

pub struct MondrianForestClassifier<F: FType> {
    trees: Vec<MondrianTreeClassifier<F>>,
    n_labels: usize,
}
impl<F: FType> MondrianForestClassifier<F> {
    pub fn new(n_trees: usize, n_features: usize, n_labels: usize) -> Self {
        let tree_default = MondrianTreeClassifier::new(n_features, n_labels);
        let trees = vec![tree_default; n_trees];
        MondrianForestClassifier::<F> { trees, n_labels }
    }

    /// Function in River is "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<F>, y: usize) {
        for tree in &mut self.trees {
            tree.partial_fit(x, y);
        }
    }

    fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        let mut tot_probs = Array1::<F>::zeros(self.n_labels);
        for tree in &self.trees {
            let probs = tree.predict_proba(x);
            debug_assert!(
                !probs.iter().any(|&x| x.is_nan()),
                "Probability should not be NaN. Found: {:?}.",
                probs.to_vec()
            );
            tot_probs += &probs;
        }
        tot_probs /= F::from_usize(self.trees.len()).unwrap();
        tot_probs
    }

    pub fn score(&mut self, x: &Array1<F>, y: usize) -> F {
        let probs = self.predict_proba(x);
        let pred_idx = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        // println!("probs: {}, pred_idx: {}, y (correct): {}, is_correct: {}", probs, pred_idx, y, pred_idx == y);
        if pred_idx == y {
            F::one()
        } else {
            F::zero()
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
