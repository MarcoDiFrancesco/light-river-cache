use crate::common::{Classifier, ClfTarget, ModelTarget, Observation, RegTarget, Regressor};
use crate::mondrian_forest::alias::FType;
use crate::mondrian_forest::mondrian_tree_clf::MondrianTreeClassifier;
use ndarray::Array1; // Importing the required traits

use num::{Float, FromPrimitive};
use rand::prelude::*;

use std::collections::HashMap;

use std::usize;

use super::mondrian_tree_reg::MondrianTreeRegressor;

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

    // ######################

    pub fn cache_sort(&mut self) {
        for tree in &mut self.trees {
            tree.cache_sort();
        }
    }

    pub fn get_forest_size(&self) -> usize {
        let sizes: Vec<usize> = self.trees.iter().map(|t| t.get_tree_size()).collect();
        sizes.iter().sum()
    }

    pub fn get_forest_depth(&self) -> (f32, f32, f32, f32, f32) {
        let mut node_counts = vec![];
        let mut optimals = vec![];
        let mut avgs = vec![];
        let mut avgs_w = vec![];
        let mut maxs = vec![];
        for t in &self.trees {
            let (node_count, opt, avg, avg_w, max) = t.get_tree_depths();
            node_counts.push(node_count);
            optimals.push(opt);
            avgs.push(avg);
            avgs_w.push(avg_w);
            maxs.push(max);
        }
        // Average over #trees
        let node_n_avg = node_counts.iter().sum::<f32>() / node_counts.len() as f32;
        let opt = optimals.iter().sum::<f32>() / optimals.len() as f32;
        let avg = avgs.iter().sum::<f32>() / avgs.len() as f32;
        let avg_w = avgs_w.iter().sum::<f32>() / avgs_w.len() as f32;
        let max = maxs.iter().sum::<f32>() / maxs.len() as f32;
        (node_n_avg, opt, avg, avg_w, max)
    }

    pub fn get_sorted_count(&mut self) -> (f32, f32) {
        let mut sorted_counts = vec![];
        let mut unsorted_counts = vec![];
        for t in &mut self.trees {
            let (sorted_c, unsorted_c) = t.get_sorted_count();
            sorted_counts.push(sorted_c);
            unsorted_counts.push(unsorted_c);
        }
        let count_tot = sorted_counts.iter().sum::<usize>() + unsorted_counts.iter().sum::<usize>();
        let sorted_avg = sorted_counts.iter().sum::<usize>() as f32 / count_tot as f32;
        let unsorted_avg = unsorted_counts.iter().sum::<usize>() as f32 / count_tot as f32;
        (sorted_avg, unsorted_avg)
    }
}

impl<F: FType> Classifier<F> for MondrianForestClassifier<F> {
    fn learn_one(&mut self, x: &Array1<F>, y: &ClfTarget) {
        for tree in &mut self.trees {
            tree.learn_one(x, y);
        }
    }

    // TODO: rename function to "predict_proba" both in trait and here
    fn predict_one(&mut self, x: &Array1<F>, y: &ClfTarget) -> F {
        let y = (*y).clone().into();
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

    fn predict_proba(&mut self, x: &Array1<F>) -> Array1<F> {
        let mut tot_probs = Array1::<F>::zeros(self.n_labels);
        for tree in &mut self.trees {
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
}

pub struct MondrianForestRegressor<F: FType> {
    trees: Vec<MondrianTreeRegressor<F>>,
}

impl<F: FType> MondrianForestRegressor<F> {
    pub fn new(n_trees: usize, n_features: usize) -> Self {
        let tree_default = MondrianTreeRegressor::new(n_features);
        let trees = vec![tree_default; n_trees];
        MondrianForestRegressor::<F> { trees }
    }

    // ######################

    pub fn cache_sort(&mut self) {
        for tree in &mut self.trees {
            tree.cache_sort();
        }
    }

    pub fn get_forest_size(&self) -> usize {
        let sizes: Vec<usize> = self.trees.iter().map(|t| t.get_tree_size()).collect();
        sizes.iter().sum()
    }

    pub fn get_forest_depth(&self) -> (f32, f32, f32, f32, f32) {
        let mut node_counts = vec![];
        let mut optimals = vec![];
        let mut avgs = vec![];
        let mut avgs_w = vec![];
        let mut maxs = vec![];
        for t in &self.trees {
            let (node_count, opt, avg, avg_w, max) = t.get_tree_depths();
            node_counts.push(node_count);
            optimals.push(opt);
            avgs.push(avg);
            avgs_w.push(avg_w);
            maxs.push(max);
        }
        // Average over #trees
        let node_n_avg = node_counts.iter().sum::<f32>() / node_counts.len() as f32;
        let opt = optimals.iter().sum::<f32>() / optimals.len() as f32;
        let avg = avgs.iter().sum::<f32>() / avgs.len() as f32;
        let avg_w = avgs_w.iter().sum::<f32>() / avgs_w.len() as f32;
        let max = maxs.iter().sum::<f32>() / maxs.len() as f32;
        (node_n_avg, opt, avg, avg_w, max)
    }

    pub fn get_sorted_count(&mut self) -> (f32, f32) {
        let mut sorted_counts = vec![];
        let mut unsorted_counts = vec![];
        for t in &mut self.trees {
            let (sorted_c, unsorted_c) = t.get_sorted_count();
            sorted_counts.push(sorted_c);
            unsorted_counts.push(unsorted_c);
        }
        let count_tot = sorted_counts.iter().sum::<usize>() + unsorted_counts.iter().sum::<usize>();
        let sorted_avg = sorted_counts.iter().sum::<usize>() as f32 / count_tot as f32;
        let unsorted_avg = unsorted_counts.iter().sum::<usize>() as f32 / count_tot as f32;
        (sorted_avg, unsorted_avg)
    }
}

impl<F: FType> Regressor<F> for MondrianForestRegressor<F> {
    fn learn_one(&mut self, x: &Array1<F>, y: &RegTarget<F>) {
        for tree in &mut self.trees {
            tree.learn_one(x, y);
        }
    }

    fn predict_one(&mut self, x: &Array1<F>, y: &RegTarget<F>) -> F {
        let mut pred_tot = F::zero();
        for tree in &mut self.trees {
            pred_tot += tree.predict_one(x, y);
        }
        pred_tot /= F::from_usize(self.trees.len()).unwrap();
        pred_tot
    }
}
