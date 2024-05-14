use crate::classification::mondrian_node::{Node, Stats};
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Exp};
use std::collections::HashSet;
use std::fmt;
use std::usize;

#[derive(Clone)]
pub struct MondrianTreeClassifier {
    n_features: usize,
    n_labels: usize,
    rng: ThreadRng,
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl fmt::Display for MondrianTreeClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n┌ MondrianTreeClassifier")?;
        self.recursive_repr(self.root, f, "│ ")
    }
}
impl MondrianTreeClassifier {
    /// Helper method to recursively format node details.
    fn recursive_repr(
        &self,
        node_idx: Option<usize>,
        f: &mut fmt::Formatter<'_>,
        prefix: &str,
    ) -> fmt::Result {
        if let Some(idx) = node_idx {
            let node = &self.nodes[idx];
            writeln!(
                f,
                "{}├─Node {}: left={:?}, right={:?}, parent={:?}, time={:.3}, min={:?}, max={:?}, counts={:?}",
                prefix,
                idx,
                node.left,
                node.right,
                node.parent,
                node.time,
                node.min_list.to_vec(),
                node.max_list.to_vec(),
                node.stats.counts.to_vec(),
            )?;
            // writeln!(
            //     f,
            //     "{}├─Node {}: left={:?}, right={:?}, parent={:?}, time={:.3}, counts={:?}",
            //     prefix,
            //     idx,
            //     node.left,
            //     node.right,
            //     node.parent,
            //     node.time,
            //     node.stats.counts.to_vec(),
            // )?;

            self.recursive_repr(node.left, f, &(prefix.to_owned() + "│ "))?;
            self.recursive_repr(node.right, f, &(prefix.to_owned() + "│ "))?;
        }
        Ok(())
    }
}

impl MondrianTreeClassifier {
    pub fn new(n_features: usize, n_labels: usize) -> Self {
        MondrianTreeClassifier {
            n_features,
            n_labels,
            rng: rand::thread_rng(),
            nodes: vec![],
            root: None,
        }
    }

    fn create_leaf(
        &mut self,
        x: &Array1<f32>,
        y: usize,
        parent: Option<usize>,
        time: f32,
    ) -> usize {
        let mut node = Node {
            parent,
            time, // f32::from(1e9).unwrap(), // Very large value
            is_leaf: true,
            min_list: x.clone(),
            max_list: x.clone(),
            feature: 0,
            threshold: 0.0,
            left: None,
            right: None,
            stats: Stats::new(self.n_labels, self.n_features),
        };

        node.add_to_leaf(x, y);
        self.nodes.push(node);
        let node_idx = self.nodes.len() - 1;
        node_idx
    }

    fn test_tree(&self) {
        // TODO: move to test
        for node_idx in 0..self.nodes.len() {
            // TODO: check if self.root is None, if so tree should be empty
            if node_idx == self.root.unwrap() {
                // Root node
                assert!(self.nodes[node_idx].parent.is_none(), "Root has a parent.");
            } else {
                // Non-root node
                assert!(
                    !self.nodes[node_idx].parent.is_none(),
                    "Non-root node has no parent"
                )
            }
        }

        let children_l: Vec<usize> = self.nodes.iter().filter_map(|node| node.left).collect();
        let children_r: Vec<usize> = self.nodes.iter().filter_map(|node| node.right).collect();
        let children = [children_l.clone(), children_r.clone()].concat();
        let mut seen = HashSet::new();
        let has_duplicates = children.iter().any(|item| !seen.insert(item));
        assert!(
            !has_duplicates,
            "Multiple nodes share one child. Children left: {:?}, Children right: {:?}",
            children_l, children_r
        );
    }

    fn compute_split_time(
        &self,
        time: f32,
        exp_sample: f32,
        node_idx: usize,
        y: usize,
        extensions_sum: f32,
    ) -> f32 {
        if self.nodes[node_idx].is_dirac(y) {
            // println!(
            //     "go_downwards() - node: {node_idx} - extensions_sum: {:?} - all same class",
            //     extensions_sum
            // );
            return 0.0;
        }

        if extensions_sum > 0.0 {
            let split_time = time + exp_sample;

            // From River: If the node is a leaf we must split it
            if self.nodes[node_idx].is_leaf {
                // println!(
                //     "go_downwards() - node: {node_idx} - extensions_sum: {:?} - split is_leaf",
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
                //     "go_downwards() - node: {node_idx} - extensions_sum: {:?} - split mid tree",
                //     extensions_sum
                // );
                return split_time;
            }
            // println!("go_downwards() - node: {node_idx} - extensions_sum: {:?} - not increased enough to split (mid node)", extensions_sum);
        } else {
            // println!(
            //     "go_downwards() - node: {node_idx} - extensions_sum: {:?} - not outside box",
            //     extensions_sum
            // );
        }

        0.0
    }

    fn go_downwards(&mut self, node_idx: usize, x: &Array1<f32>, y: usize) -> usize {
        let time = self.nodes[node_idx].time;
        let node_min_list = &self.nodes[node_idx].min_list;
        let node_max_list = &self.nodes[node_idx].max_list;
        let extensions = {
            let e_min = (node_min_list - x).mapv(|v| f32::max(v, 0.0));
            let e_max = (x - node_max_list).mapv(|v| f32::max(v, 0.0));
            &e_min + &e_max
        };
        // 'T' in River
        let exp_sample = {
            let lambda = extensions.sum();
            let exp_dist = Exp::new(lambda).unwrap();
            let exp_sample = exp_dist.sample(&mut self.rng);
            // DEBUG: shadowing with Exp expected value
            let exp_sample = 1.0 / lambda;
            exp_sample
        };
        let split_time = self.compute_split_time(time, exp_sample, node_idx, y, extensions.sum());
        if split_time > 0.0 {
            // Here split the current node: if leaf we add children, otherwise
            // we add a new node along the path
            let feature = {
                let cumsum = extensions
                    .iter()
                    .scan(0.0, |acc, &x| {
                        *acc = *acc + x;
                        Some(*acc)
                    })
                    .collect::<Array1<f32>>();
                let e_sample = self.rng.gen::<f32>() * extensions.sum();
                // DEBUG: shadowing with expected value
                let e_sample = 0.5 * extensions.sum();
                cumsum.iter().position(|&val| val > e_sample).unwrap()
            };

            let (lower_bound, upper_bound) = if x[feature] > node_min_list[feature] {
                (node_min_list[feature], x[feature])
            } else {
                (x[feature], node_max_list[feature])
            };
            let threshold = self.rng.gen_range(lower_bound..upper_bound);
            // DEBUG: split in the middle
            let threshold = (lower_bound + upper_bound) / 2.0;

            let mut min_list = node_min_list.clone();
            let mut max_list = node_max_list.clone();
            min_list.zip_mut_with(x, |a, &b| *a = f32::min(*a, b));
            max_list.zip_mut_with(x, |a, &b| *a = f32::max(*a, b));

            // Create and push new parent node
            let parent_node = Node {
                parent: self.nodes[node_idx].parent,
                time: self.nodes[node_idx].time,
                is_leaf: false,
                min_list,
                max_list,
                feature,
                threshold,
                left: None,
                right: None,
                stats: Stats::new(self.n_labels, self.n_features),
            };

            self.nodes.push(parent_node);
            let parent_idx = self.nodes.len() - 1;
            let sibling_idx = self.create_leaf(x, y, Some(parent_idx), split_time);

            // Set the children appropriately
            if x[feature] <= threshold {
                // Grandpa: self.nodes[node_idx].parent
                // (new) Parent: parent_idx
                // Child: node_idx
                // (new) Sibling: sibling_idx
                self.nodes[parent_idx].left = Some(sibling_idx);
                self.nodes[parent_idx].right = Some(node_idx);
            } else {
                self.nodes[parent_idx].left = Some(node_idx);
                self.nodes[parent_idx].right = Some(sibling_idx);
            }

            self.nodes[node_idx].parent = Some(parent_idx);
            self.nodes[node_idx].time = split_time;

            self.update_downwards(parent_idx);

            return parent_idx;
        } else {
            // No split, just update the node. If leaf add to count, else call recursively next child node.

            let node = &mut self.nodes[node_idx];
            // println!("pre - node: {:?}, node range: ({:?}-{:?}), x: {:?}", node_idx, node.min_list.to_vec(), node.max_list.to_vec(), x.to_vec());
            node.min_list.zip_mut_with(x, |a, b| *a = f32::min(*a, *b));
            node.max_list.zip_mut_with(x, |a, b| *a = f32::max(*a, *b));
            // println!("post- node: {:?}, node range: ({:?}-{:?}), x: {:?}", node_idx, node.min_list.to_vec(), node.max_list.to_vec(), x.to_vec());

            if node.is_leaf {
                // println!("else - updating leaf: {node_idx}");
                node.add_to_leaf(x, y);
            } else {
                // println!("else - updating non-leaf: {node_idx}");
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
                self.update_downwards(node_idx);
            }
            return node_idx;
        }
    }

    /// Update 'node stats' by merging 'right child stats + left child stats'.
    fn update_downwards(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        let left_s = &self.nodes[node.left.unwrap()].stats;
        let right_s = &self.nodes[node.right.unwrap()].stats;
        let merge_s = node.get_stats_from_children(left_s, right_s);
        let node = &mut self.nodes[node_idx];
        node.stats = merge_s;
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one.
    ///
    /// Function in River/LightRiver: "learn_one()"
    pub fn partial_fit(&mut self, x: &Array1<f32>, y: usize) {
        self.root = match self.root {
            None => Some(self.create_leaf(x, y, None, 0.0)),
            Some(root_idx) => Some(self.go_downwards(root_idx, x, y)),
        };
        // println!("partial_fit() tree post {}", self);
    }

    fn fit(&self) {
        unimplemented!("Make the program first work with 'partial_fit', then implement this")
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(&self, x: &Array1<f32>) -> Array1<f32> {
        // println!("predict_proba() - tree size: {}", self.nodes.len());
        // self.test_tree();
        self.predict(x, self.root.unwrap(), 1.0)
    }

    fn predict(&self, x: &Array1<f32>, node_idx: usize, p_not_separated_yet: f32) -> Array1<f32> {
        let node = &self.nodes[node_idx];

        // Probability 'p' of the box not splitting.
        //     eta (box dist): larger distance, more prob of splitting
        //     d (time delta with parent): more dist with parent, more prob of splitting
        let p = {
            let d = node.time - self.get_parent_time(node_idx);
            let dist_max = (x - &node.max_list).mapv(|v| f32::max(v, 0.0));
            let dist_min = (&node.min_list - x).mapv(|v| f32::max(v, 0.0));
            let eta = dist_min.sum() + dist_max.sum();
            1.0 - (-d * eta).exp()
        };

        // Generate a result for the current node using its statistics.
        let res = node.stats.create_result(x, p_not_separated_yet * p);

        let w = p_not_separated_yet * (1.0 - p);
        if node.is_leaf {
            let res2 = node.stats.create_result(x, w);
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

    fn get_parent_time(&self, node_idx: usize) -> f32 {
        // If node is root, time is 0
        match self.nodes[node_idx].parent {
            Some(parent_idx) => self.nodes[parent_idx].time,
            None => 0.0,
        }
    }

    pub fn get_tree_size(&self) -> usize {
        self.nodes.len()
    }

    // //////////////////////////////////////////////////////////
    //  Caching
    // //////////////////////////////////////////////////////////

    /// Return 'index of candidate vector' of the candidate with highest count.
    ///
    /// TODO: check if this function is useful. It looks like we are always
    ///       returning index 0.
    ///
    /// e.g. get_node_highest_prob(candidates=[(11, 3), (1, 3), (9, 2)]) -> 0
    fn get_node_highest_prob(&self, candidates: &Vec<(usize, usize)>) -> (usize, usize) {
        let mut max_count = 0;
        let mut max_idx = 0;
        for (idx, (_, node_count)) in candidates.iter().enumerate() {
            if node_count > &max_count {
                max_count = *node_count;
                max_idx = idx;
            }
        }
        candidates[max_idx]
    }

    /// Given a starting node index i find path with highest count.
    ///
    /// e.g. considering tree below
    ///     get_node_chain(0, candidates) -> [(0, 10), (1, 7), (3, 5)]
    ///     get_node_chain(2, candidates) -> [(2, 3), (5, 2)]
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
        candidates: &mut Vec<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut set = vec![i];
        let mut node = &self.nodes[i.0];
        let mut i;
        while !node.is_leaf {
            let idx_l = node.left.unwrap();
            let idx_r = node.right.unwrap();
            let count_l = self.nodes[idx_l].stats.counts.sum();
            let count_r = self.nodes[idx_r].stats.counts.sum();
            if count_l >= count_r {
                // Add right to candidates
                candidates.push((idx_r, count_r));
                // Set i to left
                node = &self.nodes[idx_l];
                i = (idx_l, count_l);
            } else {
                candidates.push((idx_l, count_l));
                node = &self.nodes[idx_r];
                i = (idx_r, count_r);
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
        // root: (node_idx, node_count)
        let root = {
            // Order node by count in reverse order.
            // nodes_probs: [(node_idx, node_count)]
            // Example tree in docstring: [(0, 10), (1, 7), (3, 5), ...]
            let mut nodes_probs: Vec<(usize, usize)> = self
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (i, n.stats.counts.sum()))
                .collect();
            nodes_probs.sort_by_key(|k| k.1);
            nodes_probs.reverse();
            nodes_probs[0]
        };
        // A
        let mut new_order: Vec<(usize, usize)> = vec![];
        // C
        let mut candidates = vec![root];
        while !candidates.is_empty() {
            let i = self.get_node_highest_prob(&candidates);
            // Remove candidate
            candidates.remove(candidates.iter().position(|x| *x == i).unwrap());
            // S
            let set = self.get_node_chain(i, &mut candidates);
            new_order.extend(&set);

            // println!(
            //     "set: {:?},\nnew_order: {:?}\ncandidates: {:?}\n",
            //     set.to_vec(),
            //     new_order.to_vec(),
            //     candidates.to_vec(),
            // );
        }
        assert!(new_order.len() == self.nodes.len());
        // From [(0, 10), (1, 7), (3, 5), ...] to [0, 1, 3, ...]
        new_order.iter().map(|(idx, _)| *idx).collect()
    }

    /// Sort nodes in the tree by the likelihood of access. Arrange the nodes
    /// so that the most probable next node is positioned next in the vector.
    ///
    /// Follows implementation by [1] algorithm 2 'Optimized native Tree'.
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

        // Allocate ordered nodes in new vector
        let mut nodes_reordered = Vec::with_capacity(self.nodes.len());
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
            self.nodes[self.root.unwrap()].time == 0.0,
            "New order does not set root correctly. Found time of root: {}, instead of 0.",
            self.nodes[self.root.unwrap()].time
        );
        // self.test_tree();
        // println!("cache_sort() - tree post {}", self);
    }
}
