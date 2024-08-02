use crate::common::{Classifier, ClfTarget, RegTarget, Regressor};
use crate::mondrian_forest::alias::FType;
use crate::mondrian_forest::mondrian_node::{NodeClassifier, NodeRegressor};
use ndarray::{Array1, Array2};
use num::ToPrimitive;
use num::{Float, FromPrimitive};
use rand::prelude::*;
use rand_distr::{Distribution, Exp};
use std::collections::{BinaryHeap, HashSet};
use std::fmt;
use std::usize;

const MAX_NODES: usize = 100_000;

#[derive(Clone)]
pub struct MondrianTreeRegressor<F: FType> {
    n_features: usize,
    rng: ThreadRng,
    nodes: Vec<NodeRegressor<F>>,
    root: Option<usize>,
    sorted_c: usize,
    unsorted_c: usize,
    previous_node: Option<usize>,
}

impl<F: FType + fmt::Display> fmt::Display for MondrianTreeRegressor<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "")?;
        if let Some(root) = self.root {
            self.recursive_repr(Some(root), f, "", true)
        } else {
            Ok(())
        }
    }
}

impl<F: FType + fmt::Display> MondrianTreeRegressor<F> {
    fn recursive_repr(
        &self,
        node_idx: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
        is_last: bool,
    ) -> fmt::Result {
        if let Some(idx) = node_idx {
            let node = &self.nodes[idx];
            let node_prefix = if is_last { "└─" } else { "├─" };
            let child_prefix = if is_last { "  " } else { "│ " };
            let feature = if self.nodes[idx].feature > 100 {
                String::from("_")
            } else {
                self.nodes[idx].feature.to_string()
            };
            writeln!(
                f,
                // "{}{}Node {}: time={:.3}, min={:?}, max={:?}, thrs={:.2}, f={}, sums={}, counts={}",
                "{}{}Node {}: f={}, counts={}",
                prefix,
                node_prefix,
                idx,
                // node.time,
                // node.range_min.to_vec(),
                // node.range_max.to_vec(),
                // node.threshold,
                feature,
                // node.sums,
                node.counts,
            )?;

            let mut children = vec![];
            if let Some(left) = node.left {
                children.push((left, node.right.is_none())); // Left child
            }
            if let Some(right) = node.right {
                children.push((right, true)); // Right child
            }

            for (i, (child, last)) in children.into_iter().enumerate() {
                self.recursive_repr(Some(child), f, &(prefix.to_owned() + child_prefix), last)?;
            }
        }
        Ok(())
    }
}

impl<F: FType> Regressor<F> for MondrianTreeRegressor<F> {
    fn learn_one(&mut self, x: &Array1<F>, y: &RegTarget<F>) {
        self.root = match self.root {
            None => Some(self.create_leaf(x, y, None, F::zero())),
            Some(root_idx) => Some(self.go_downwards(root_idx, x, y)),
        };
    }

    fn predict_one(&mut self, x: &Array1<F>, y: &RegTarget<F>) -> F {
        // self.test_tree();
        self.previous_node = None;
        self.predict(x, self.root.unwrap(), F::one())
    }
}

impl<F: FType> MondrianTreeRegressor<F> {
    pub fn new(n_features: usize) -> Self {
        MondrianTreeRegressor::<F> {
            n_features,
            rng: rand::thread_rng(),
            nodes: Vec::with_capacity(MAX_NODES),
            root: None,
            sorted_c: 0,
            unsorted_c: 0,
            previous_node: None,
        }
    }

    fn create_leaf(
        &mut self,
        x: &Array1<F>,
        y: &RegTarget<F>,
        parent: Option<usize>,
        time: F,
    ) -> usize {
        let mut node = NodeRegressor::<F> {
            parent,
            time,
            is_leaf: true,
            range_min: x.clone(),
            range_max: x.clone(),
            feature: usize::MAX,
            threshold: F::infinity(),
            left: None,
            right: None,
            sums: F::zero(),
            counts: 0,
            n_features: self.n_features,
        };
        node.add(x, y);
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    fn create_empty_node(&mut self, parent: Option<usize>, time: F) -> usize {
        let node = NodeRegressor::<F> {
            parent,
            time,
            is_leaf: true,
            range_min: Array1::from_elem(self.n_features, F::infinity()),
            range_max: Array1::from_elem(self.n_features, -F::infinity()),
            feature: usize::MAX,
            threshold: F::infinity(),
            left: None,
            right: None,
            sums: F::zero(),
            counts: 0,
            n_features: self.n_features,
        };
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    fn test_tree(&self) {
        for node_idx in 0..self.nodes.len() {
            // TODO: check if self.root is None, if so tree should be empty
            if node_idx == self.root.unwrap() {
                // Root node
                debug_assert!(self.nodes[node_idx].parent.is_none(), "Root has a parent.");
            } else {
                // Non-root node
                debug_assert!(
                    !self.nodes[node_idx].parent.is_none(),
                    "Non-root node has no parent"
                )
            }
        }

        fn point_inside_area<F: std::cmp::PartialOrd>(
            point: &Array1<F>,
            range_min: &Array1<F>,
            range_max: &Array1<F>,
        ) -> bool {
            point.iter().zip(range_min.iter()).all(|(a, b)| *a >= *b)
                && point.iter().zip(range_max.iter()).all(|(a, b)| *a <= *b)
        }

        // Check if siblings are sharing area of the rectangle.
        //
        // e.g. Tree
        //      └─Node: min=[0, 0], max=[3, 3]
        //        ├─Node: min=[0, 0], max=[2, 2]
        //        └─Node: min=[1, 1], max=[3, 3] <----- Overlap in area [1, 1] to [2, 2]
        fn siblings_share_area<F: Float + std::cmp::PartialOrd>(
            left: &NodeRegressor<F>,
            right: &NodeRegressor<F>,
        ) -> bool {
            point_inside_area(&left.range_min, &right.range_min, &right.range_max)
                || point_inside_area(&left.range_max, &right.range_min, &right.range_max)
                || point_inside_area(&right.range_min, &left.range_min, &left.range_max)
                || point_inside_area(&right.range_max, &left.range_min, &left.range_max)
        }

        /// Check if child is inside parent's rectangle.
        ///
        /// e.g. Tree
        ///      └─Node: min=[0, 0], max=[3, 3]
        ///        ├─Node: min=[4, 4], max=[5, 5] <----- Child outside parent
        ///        └─Node: min=[1, 1], max=[2, 2]
        ///
        /// NOTE: Removed this check because River breaks this rule. In my opinion River implementation
        /// is wrong, it makes 0% sense that after a mid tree split the parent can have the range (min/max) that
        /// does not contain the child. But, I'm following the implementation 1:1 so I comment this check.
        /// It must be checked in the future.
        /// (An example: https://i.imgur.com/Yk4ZeuZ.png)
        fn child_inside_parent<F: Float + std::cmp::PartialOrd>(
            parent: &NodeRegressor<F>,
            child: &NodeRegressor<F>,
        ) -> bool {
            // Skip if child is not initialized
            if child.range_min.iter().any(|&x| x.is_infinite()) {
                return true;
            }
            // Skip if parent is not initalized. Happens with split mid tree.
            if parent.range_min.iter().any(|&x| x.is_infinite()) {
                return true;
            }
            point_inside_area(&child.range_min, &parent.range_min, &parent.range_max)
                & point_inside_area(&child.range_max, &parent.range_min, &parent.range_max)
        }

        /// Check if parent threshold cuts child.
        ///
        /// e.g. Tree
        ///      └─Node: min=[0, 0], max=[3, 3], thrs=0.5, f=0
        ///        ├─Node: min=[0, 0], max=[1, 1], thrs=inf, f=_ <----- Threshold (0.5) cuts child
        ///        └─Node: min=[2, 2], max=[3, 3], thrs=inf, f=_
        fn threshold_cuts_child<F: Float + std::cmp::PartialOrd>(
            parent: &NodeRegressor<F>,
            child: &NodeRegressor<F>,
        ) -> bool {
            // Skip if child is not initialized
            if child.range_min.iter().any(|&x| x.is_infinite()) {
                return false;
            }
            let thrs = parent.threshold;
            let f = parent.feature;
            (child.range_min[f] < thrs) & (thrs < child.range_max[f])
        }

        /// Check if left child is on left of the threshold, right child on right of it.
        ///
        /// e.g. Tree
        ///      └─Node: min=[0, 0], max=[4, 4], thrs=2, f=0
        ///        ├─Node: min=[0, 0], max=[0, 0], thrs=inf, f=_
        ///        └─Node: min=[1, 1], max=[1, 1], thrs=inf, f=_ <----- Right child on found in the left of the threshold
        fn children_on_correct_side<F: Float + std::cmp::PartialOrd>(
            parent: &NodeRegressor<F>,
            left: &NodeRegressor<F>,
            right: &NodeRegressor<F>,
        ) -> bool {
            let thrs = parent.threshold;
            let f = parent.feature;
            // Left node
            if !left.range_min.iter().any(|&x| x.is_infinite()) {
                if (left.range_min[f] > thrs) | (left.range_max[f] > thrs) {
                    println!("Left node right to the threshold");
                    return false;
                }
            }
            // Right node
            if !right.range_min.iter().any(|&x| x.is_infinite()) {
                if (right.range_min[f] < thrs) | (right.range_max[f] < thrs) {
                    return false;
                }
            }
            return true;
        }

        /// Checking if parent count is the sum of the children
        ///
        /// e.g. Tree
        ///      └─Node: counts=[0, 2, 1] <---- Error: counts sould be [0, 2, 2]
        ///        ├─Node: counts=[0, 1, 2]
        ///        └─Node: counts=[0, 1, 0]
        ///
        /// NOTE: Commented since this in River this assumption is violated.
        /// It happens after adding leaves.
        /// e.g. River output of a tree:
        /// ┌ Node: counts=[0, 2, 4]
        /// │ ├─ Node: counts=[0, 0, 4] <---- This is the sum of the children
        /// │ │ ├─ Node: counts=[0, 0, 4]
        /// │ │ ├─ Node: counts=[0, 0, 0]
        /// │ ├─ Node: counts=[0, 1, 0]
        /// Becomes after one sample:
        /// ┌ Node: counts=[1, 2, 4]
        /// │ ├─ Node: counts=[1, 0, 4] <---- Not the sum of the children anymore
        /// │ │ ├─ Node: counts=[1, 0, 0]
        /// │ │ │ ├─ Node: counts=[1, 0, 0]
        /// │ │ │ ├─ Node: counts=[0, 0, 0]
        /// │ │ ├─ Node: counts=[0, 0, 0]
        /// │ ├─ Node: counts=[0, 1, 0]
        fn parent_has_sibling_counts<F: Float + std::cmp::PartialOrd>(
            parent: &NodeRegressor<F>,
            left: &NodeRegressor<F>,
            right: &NodeRegressor<F>,
        ) -> bool {
            (&left.counts + &right.counts) == parent.counts
        }

        /// Checking if parent (with children) does not have infinite values.
        ///
        /// e.g. Tree
        ///      └─Node: min=[0.2, 0.1], max=[2.6, 1.6]
        ///        ├─Node: min=[inf, inf], max=[-inf, -inf] <----- This is not possible
        ///        │ ├─Node: min=[0.3, 0.2], max=[0.3, 0.2]
        ///        │ └─Node: min=[0.7, 0.6], max=[0.8, 0.9]
        ///        └─Node: min=[2.6, 1.6], max=[2.6, 1.6]
        fn parent_has_finite_values<F: Float + std::cmp::PartialOrd>(
            parent: &NodeRegressor<F>,
            left: &NodeRegressor<F>,
            right: &NodeRegressor<F>,
        ) -> bool {
            parent.range_min.iter().all(|&x| x.is_finite())
                && parent.range_max.iter().all(|&x| x.is_finite())
        }

        for node_idx in 0..self.nodes.len() {
            let node = &self.nodes[node_idx];
            if node.left.is_some() {
                let left_idx = node.left.unwrap();
                let right_idx = node.right.unwrap();
                let left = &self.nodes[left_idx];
                let right = &self.nodes[right_idx];

                // Siblings share area
                debug_assert!(
                    !siblings_share_area(left, right),
                    "Siblings share area. \nNode {}: {}, \nNode {}: {}\nTree{}",
                    left_idx,
                    left,
                    right_idx,
                    right,
                    self
                );

                // Child inside parent
                debug_assert!(
                    child_inside_parent(node, left),
                    "Left child outiside parent area. \nParent {}: {}, \nChild {}: {}\nTree{}",
                    node_idx,
                    node,
                    left_idx,
                    left,
                    self
                );
                debug_assert!(
                    child_inside_parent(node, right),
                    "Right child outiside parent area. \nParent {}: {}, \nChild {}: {}\nTree{}",
                    node_idx,
                    node,
                    right_idx,
                    right,
                    self
                );

                // Threshold cuts child
                debug_assert!(
                    !threshold_cuts_child(node, right),
                    "Threshold (of the parent) cuts child. \nParent {}: {}, \nChild {}: {}\nTree{}",
                    node_idx,
                    node,
                    right_idx,
                    right,
                    self
                );
                debug_assert!(
                    !threshold_cuts_child(node, left),
                    "Threshold (of the parent) cuts child. \nParent {}: {}, \nChild {}: {}\nTree{}",
                    node_idx,
                    node,
                    left_idx,
                    left,
                    self
                );

                // Child on correct side of the threshold
                debug_assert!(
                    children_on_correct_side(node, left, right),
                    "One child is on the wrong side of the split. \nThreshold (Parent {}): {}, Left: {}, Right: {}\nTree{}",
                    node_idx,
                    node.threshold,
                    left_idx,
                    right_idx,
                    self
                );

                // Parent count has sibling count sum
                // debug_assert!(
                //     parent_has_sibling_counts(node, left, right),
                //     "Parent count is not sibling's sum. \nNode {}: {}\nTree{}",
                //     node_idx,
                //     node,
                //     self
                // );

                // Parent does not have infinite values
                debug_assert!(
                    parent_has_finite_values(node, left, right),
                    "Parent should not have infinite values. \nNode {}: {}\nTree{}",
                    node_idx,
                    node,
                    self
                );
            }
        }

        // Check if parents are not sharing children
        let children_l: Vec<usize> = self.nodes.iter().filter_map(|node| node.left).collect();
        let children_r: Vec<usize> = self.nodes.iter().filter_map(|node| node.right).collect();
        let children = [children_l.clone(), children_r.clone()].concat();
        let mut seen = HashSet::new();
        let has_duplicates = children.iter().any(|item| !seen.insert(item));
        debug_assert!(
            !has_duplicates,
            "Multiple nodes share one child. Children left: {:?}, Children right: {:?}",
            children_l, children_r
        );
    }

    fn compute_split_time(&self, time: F, exp_sample: F, node_idx: usize, extensions_sum: F) -> F {
        // If we are in a non-initialized leaf
        if extensions_sum.is_infinite() {
            return F::zero();
        }

        if extensions_sum > F::zero() {
            let split_time = time + exp_sample;

            // From River: If the node is a leaf we must split it
            if self.nodes[node_idx].is_leaf {
                // println!(
                //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - split is_leaf",
                //     extensions_sum
                // );
                return split_time;
            }

            // From River: Otherwise we apply Mondrian process dark magic :)
            // 1. We get the creation time of the childs (left and right is the same)
            let child_idx = self.nodes[node_idx].left.unwrap();
            let child_time = self.nodes[child_idx].time;
            // 2. We check if splitting time occurs before child creation time
            if split_time < child_time {
                // println!(
                //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - split mid tree",
                //     extensions_sum
                // );
                return split_time;
            }
            // println!("compute_split_time() - node: {node_idx} - extensions_sum: {:?} - not increased enough to split (mid node)", extensions_sum);
        } else {
            // println!(
            //     "compute_split_time() - node: {node_idx} - extensions_sum: {:?} - not outside box",
            //     extensions_sum
            // );
        }

        F::zero()
    }

    fn go_downwards(&mut self, node_idx: usize, x: &Array1<F>, y: &RegTarget<F>) -> usize {
        let time = self.nodes[node_idx].time;
        let node_range_min = &self.nodes[node_idx].range_min;
        let node_range_max = &self.nodes[node_idx].range_max;
        let extensions = {
            let e_min = (node_range_min - x).mapv(|v| F::max(v, F::zero()));
            let e_max = (x - node_range_max).mapv(|v| F::max(v, F::zero()));
            &e_min + &e_max
        };
        // 'T' in River
        let exp_sample = {
            let lambda = extensions.sum();
            let exp_dist = Exp::new(lambda.to_f32().unwrap()).unwrap();
            let exp_sample = F::from_f32(exp_dist.sample(&mut self.rng)).unwrap();
            // DEBUG: shadowing with Exp expected value
            let exp_sample = F::one() / lambda;
            exp_sample
        };
        let split_time = self.compute_split_time(time, exp_sample, node_idx, extensions.sum());
        if split_time > F::zero() {
            // Split the current node: if leaf we two leafs, otherwise
            // we add a new node along the path
            let feature = {
                let cumsum = extensions
                    .iter()
                    .scan(F::zero(), |acc, &x| {
                        *acc = *acc + x;
                        Some(*acc)
                    })
                    .collect::<Array1<F>>();
                let e_sample = F::from_f32(self.rng.gen::<f32>()).unwrap() * extensions.sum();
                // DEBUG: shadowing with expected value
                let e_sample = F::from_f32(0.5).unwrap() * extensions.sum();
                cumsum.iter().position(|&val| val > e_sample).unwrap()
            };

            let mut is_right_extension = x[feature] > node_range_min[feature];

            let (lower_bound, upper_bound) = if is_right_extension {
                (
                    node_range_max[feature].to_f32().unwrap(),
                    x[feature].to_f32().unwrap(),
                )
            } else {
                (
                    x[feature].to_f32().unwrap(),
                    node_range_min[feature].to_f32().unwrap(),
                )
            };
            let threshold: F = F::from_f32(self.rng.gen_range(lower_bound..upper_bound)).unwrap();
            // DEBUG: split in the middle
            let threshold = F::from_f32((lower_bound + upper_bound) / 2.0).unwrap();

            let mut range_min = node_range_min.clone();
            let mut range_max = node_range_max.clone();
            range_min.zip_mut_with(x, |a, &b| *a = F::min(*a, b));
            range_max.zip_mut_with(x, |a, &b| *a = F::max(*a, b));

            if self.nodes[node_idx].is_leaf {
                // Add two leaves.
                // println!("go_downwards() - split_time > 0 (is leaf)");
                let leaf_full = self.create_leaf(x, y, Some(node_idx), split_time);
                let leaf_empty = self.create_empty_node(Some(node_idx), split_time);
                // if x[feature] <= threshold {
                if is_right_extension {
                    self.nodes[node_idx].left = Some(leaf_empty);
                    self.nodes[node_idx].right = Some(leaf_full);
                } else {
                    self.nodes[node_idx].left = Some(leaf_full);
                    self.nodes[node_idx].right = Some(leaf_empty);
                }
                self.nodes[node_idx].range_min = range_min;
                self.nodes[node_idx].range_max = range_max;
                self.nodes[node_idx].threshold = threshold;
                self.nodes[node_idx].feature = feature;
                self.nodes[node_idx].is_leaf = false;
                self.update_downwards(node_idx);
                return node_idx;
            } else {
                // Add node along the path.
                // println!("go_downwards() - split_time > 0 (not leaf)");
                let parent_node = NodeRegressor {
                    parent: self.nodes[node_idx].parent,
                    time: self.nodes[node_idx].time,
                    is_leaf: false,
                    range_min,
                    range_max,
                    feature,
                    threshold,
                    left: None,
                    right: None,
                    sums: F::zero(),
                    counts: 0,
                    n_features: self.n_features,
                };
                self.nodes.push(parent_node);
                let parent_idx = self.nodes.len() - 1;

                let sibling_idx = self.create_leaf(x, y, Some(parent_idx), split_time);
                // println!(
                //     "grandpa: {:?}, parent: {:?}, child: {:?}, sibling: {:?}",
                //     self.nodes[node_idx].parent, parent_idx, node_idx, sibling_idx
                // );
                // Node 1. Grandpa: self.nodes[node_idx].parent
                // └─Node 3. (new) Parent: parent_idx
                //   ├─Node 2. Child: node_idx
                //   └─Node 4. (new) Sibling: sibling_idx
                if is_right_extension {
                    self.nodes[parent_idx].left = Some(node_idx);
                    self.nodes[parent_idx].right = Some(sibling_idx);
                } else {
                    self.nodes[parent_idx].left = Some(sibling_idx);
                    self.nodes[parent_idx].right = Some(node_idx);
                }
                // self.nodes[parent_idx].stats = self.nodes[node_idx].stats.clone();
                let node = &self.nodes[node_idx].clone();
                self.nodes[parent_idx].copy_stats_from(node);
                self.nodes[node_idx].parent = Some(parent_idx);
                self.nodes[node_idx].time = split_time;

                if self.nodes[node_idx].is_leaf {
                    self.nodes[node_idx].range_min =
                        Array1::from_elem(self.n_features, F::infinity());
                    self.nodes[node_idx].range_max =
                        Array1::from_elem(self.n_features, -F::infinity());

                    // Resetting stats
                    // self.nodes[node_idx].stats = Stats::new(self.n_labels, self.n_features);
                    self.nodes[node_idx].reset_stats();
                }
                // self.update_downwards(parent_idx);
                self.nodes[parent_idx].add(x, y);
                return parent_idx;
            }
        } else {
            // No split, just update the node. If leaf add to count, else call recursively next child node.

            let node = &mut self.nodes[node_idx];
            node.range_min.zip_mut_with(x, |a, b| *a = F::min(*a, *b));
            node.range_max.zip_mut_with(x, |a, b| *a = F::max(*a, *b));

            if node.is_leaf {
                // println!("go_downwards() - split_time < 0 (is leaf)");
                node.add(x, y);
            } else {
                // println!("go_downwards() - split_time < 0 (not leaf)");
                if x[node.feature] <= node.threshold {
                    let node_left = node.left.unwrap();
                    let node_left_new = Some(self.go_downwards(node_left, x, y));
                    let node = &mut self.nodes[node_idx];
                    node.left = node_left_new;
                } else {
                    let node_right = node.right.unwrap();
                    let node_right_new = Some(self.go_downwards(node_right, x, y));
                    let node = &mut self.nodes[node_idx];
                    node.right = node_right_new;
                };
                // self.update_downwards(node_idx);
                self.nodes[node_idx].add(x, y);
            }
            return node_idx;
        }
    }

    /// Update 'node stats' by merging 'right child stats + left child stats'.
    fn update_downwards(&mut self, node_idx: usize) {
        // From River:
        // Ranges (range_min, range_max) are already up to date

        // println!("update_downwards() - node_idx: {}", node_idx);

        let node = &self.nodes[node_idx];
        let left = self.nodes[node.left.unwrap()].clone();
        let right = self.nodes[node.right.unwrap()].clone();
        self.nodes[node_idx].update_internal(left.clone(), right.clone());
    }

    fn predict(&mut self, x: &Array1<F>, node_idx: usize, p_not_separated_yet: F) -> F {
        // Sorted count
        if self.previous_node.is_some() && node_idx != 0 {
            if self.previous_node.unwrap() == node_idx - 1 {
                self.sorted_c += 1;
            } else {
                self.unsorted_c += 1;
            }
        }
        self.previous_node = Some(node_idx);

        let node = &self.nodes[node_idx];
        // Probability 'p' of the box not splitting.
        //     eta (box dist): larger distance, more prob of splitting
        //     d (time delta with parent): more dist with parent, more prob of splitting
        let p = {
            let d = node.time - self.get_parent_time(node_idx);
            let dist_max = (x - &node.range_max).mapv(|v| F::max(v, F::zero()));
            let dist_min = (&node.range_min - x).mapv(|v| F::max(v, F::zero()));
            let eta = dist_min.sum() + dist_max.sum();
            F::one() - (-d * eta).exp()
        };
        debug_assert!(!p.is_nan(), "Found probability of splitting NaN. This is probably because range_max and range_min are [inf, inf].");

        // Generate a result for the current node using its statistics.
        let res = node.create_result(x, p_not_separated_yet * p);
        let w = p_not_separated_yet * (F::one() - p);
        if node.is_leaf {
            let res2 = node.create_result(x, w);
            return res + res2;
        } else {
            let child_idx = if x[node.feature] <= node.threshold {
                node.left
            } else {
                node.right
            };
            let child_res = self.predict(x, child_idx.unwrap(), w);
            return res + child_res;
        }
    }

    fn get_parent_time(&self, node_idx: usize) -> F {
        // If node is root, time is 0
        match self.nodes[node_idx].parent {
            Some(parent_idx) => self.nodes[parent_idx].time,
            None => F::from_f32(0.0).unwrap(),
        }
    }

    // //////////////////////////////////////////////////////////
    //  Caching count
    // //////////////////////////////////////////////////////////

    pub fn get_tree_size(&self) -> usize {
        self.nodes.len()
    }

    /// Returns:
    /// - Optimal: log2 of #nodes upper bound
    /// - Average: average depth of nodes
    /// - Average weighted: ...
    /// - Max: maximum depth of nodes
    pub fn get_tree_depths(&self) -> (f32, f32, f32, f32, f32) {
        // Vector to store the depth of each node
        let mut depths = Vec::with_capacity(self.nodes.len());
        let mut nodes_w = Vec::with_capacity(self.nodes.len());

        // Function to recursively compute the depth of a node
        fn compute_depth<F: FType>(
            nodes: &Vec<NodeRegressor<F>>,
            root: &NodeRegressor<F>,
            node_idx: Option<usize>,
            current_depth: usize,
            depths: &mut Vec<usize>,
            nodes_w: &mut Vec<f32>,
        ) {
            if let Some(idx) = node_idx {
                let node = &nodes[idx];
                if node.is_leaf {
                    depths.push(current_depth);
                    let weight = node.counts as f32 / root.counts as f32;
                    nodes_w.push(current_depth as f32 * weight);
                }
                compute_depth(nodes, root, node.left, current_depth + 1, depths, nodes_w);
                compute_depth(nodes, root, node.right, current_depth + 1, depths, nodes_w);
            }
        }

        let root = &self.nodes[self.root.unwrap()];
        // Compute depths starting from the root
        compute_depth(&self.nodes, root, self.root, 0, &mut depths, &mut nodes_w);

        let node_count = self.nodes.len() as f32;

        let optim = self.nodes.len().to_f32().unwrap().log2().ceil();

        let avg_depth = depths.iter().sum::<usize>() as f32 / depths.len() as f32;

        let depths_f: Vec<f32> = depths.iter().map(|&x| x as f32).collect();
        let w_depths = Array1::from_vec(depths_f) * Array1::from_vec(nodes_w);
        let avg_w_depth = w_depths.sum();

        let max_depth = *depths.iter().max().unwrap_or(&0);
        (node_count, optim, avg_depth, avg_w_depth, max_depth as f32)
    }

    pub fn get_sorted_count(&mut self) -> (usize, usize) {
        let res = (self.sorted_c, self.unsorted_c);
        self.sorted_c = 0;
        self.unsorted_c = 0;
        res
    }

    // //////////////////////////////////////////////////////////
    //  Caching sort
    // //////////////////////////////////////////////////////////

    /// Given a starting node index i find path with highest count.
    ///
    /// e.g. considering tree below
    ///     get_node_chain(0, candidates) -> [(10, 0), (7, 1), (5, 3)]
    ///     get_node_chain(2, candidates) -> [(3, 2), (2, 5)]
    ///
    /// ┌─ Node 0: count=10
    /// │  ├─ Node 1: count=7
    /// │  │  ├─ Node 3: count=5
    /// │  │  └─ Node 4: count=2
    /// │  └─ Node 2: count=3
    /// │     ├─ Node 5: count=2
    /// │     └─ Node 6: count=1
    fn get_node_chain(
        &self,
        i: (usize, usize),
        candidates: &mut BinaryHeap<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut set = vec![i];
        let mut node = &self.nodes[i.1];
        let mut i;
        while !node.is_leaf {
            let idx_l = node.left.unwrap();
            let idx_r = node.right.unwrap();
            let count_l = self.nodes[idx_l].counts;
            let count_r = self.nodes[idx_r].counts;
            if count_l >= count_r {
                candidates.push((count_r, idx_r));
                i = (count_l, idx_l);
                node = &self.nodes[idx_l];
            } else {
                candidates.push((count_l, idx_l));
                i = (count_r, idx_r);
                node = &self.nodes[idx_r];
            };
            set.push(i);
        }
        set
    }

    /// Returns e.g. [0, 1, 3, 2, 5, 4, 6]
    ///
    /// ┌─ Node 0: count=10
    /// │  ├─ Node 1: count=7
    /// │  │  ├─ Node 3: count=5
    /// │  │  └─ Node 4: count=2
    /// │  └─ Node 2: count=3
    /// │     ├─ Node 5: count=2
    /// │     └─ Node 6: count=1
    fn get_optimized_tree_order(&self) -> Vec<usize> {
        // root: (node_count, node_idx)
        let root = {
            // Order node by count in reverse order.
            // nodes_probs: [(node_count, node_idx)]
            // Example tree in docstring: [(10, 0), (7, 1), (5, 3), ...]
            let mut nodes_probs: Vec<(usize, usize)> = self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.counts, i))
                .collect();
            nodes_probs.sort_by_key(|k| k.0);
            nodes_probs.reverse();
            nodes_probs[0]
        };
        // A: All nodes in optimized order
        let mut new_order: Vec<(usize, usize)> = vec![];
        // C: Candidates
        let mut candidates = BinaryHeap::new();
        candidates.push(root);

        while let Some(_) = candidates.peek() {
            let i = candidates.pop().unwrap();
            // S
            let set = self.get_node_chain(i, &mut candidates);
            new_order.extend(&set);
        }
        assert!(new_order.len() == self.nodes.len());
        new_order.iter().map(|(_, idx)| *idx).collect()
    }

    /// Sort nodes in the tree by the likelihood of access. Arrange the nodes
    /// so that the most probable next node is positioned next in the vector.
    ///
    /// Follows implementation by [1] Algorithm 2 'Optimized native Tree'.
    ///
    /// [1] Chen et al. (2022). Efficient Realization of Decision Trees for Real-Time Inference.
    ///     ACM Transactions on Embedded Computing Systems, 21(6), 1–26. https://doi.org/10.1145/3508019
    ///
    /// Input:
    /// ┌─ Node 2: count=10
    /// │  ├─ Node 5: count=7
    /// │  │  ├─ Node 4: count=5
    /// │  │  └─ Node 6: count=2
    /// │  └─ Node 0: count=3
    /// │     ├─ Node 1: count=2
    /// │     └─ Node 3: count=1
    /// Output:
    /// ┌─ Node 0: count=10
    /// │  ├─ Node 1: count=7
    /// │  │  ├─ Node 2: count=5
    /// │  │  └─ Node 5: count=2
    /// │  └─ Node 3: count=3
    /// │     ├─ Node 4: count=2
    /// │     └─ Node 6: count=1
    pub fn cache_sort(&mut self) {
        // println!("cache_sort() - tree: {}", self);
        let new_order: Vec<usize> = self.get_optimized_tree_order();

        // e.g. new_order=[2, 0, 1] -> [2]=0, [0]=1, [1]=2 -> new_order_mapped=[1, 2, 0]
        let mut new_order_mapped = vec![0; new_order.len()];
        for (index, &value) in new_order.iter().enumerate() {
            new_order_mapped[value] = index;
        }

        // Allocate ordered nodes in new vector. Do not set vector with capacity self.nodes.len(),
        // this would make the following pushed nodes to reallocate the vector. Talking by experience :)
        let mut nodes_reordered = Vec::with_capacity(MAX_NODES);
        for i in &new_order {
            let mut new_node = self.nodes[*i].clone();
            new_node.right = new_node.right.map(|v| new_order_mapped[v]);
            new_node.left = new_node.left.map(|v| new_order_mapped[v]);
            new_node.parent = new_node.parent.map(|v| new_order_mapped[v]);
            nodes_reordered.push(new_node);
        }
        self.nodes = nodes_reordered;

        // Change root
        self.root = Some(0);
        assert!(
            self.nodes[self.root.unwrap()].time == F::zero(),
            "New order does not set root correctly. Found time of root: {}, instead of 0.",
            self.nodes[self.root.unwrap()].time
        );
        // self.test_tree();
        // println!("cache_sort() - tree post {}", self);
    }
}
