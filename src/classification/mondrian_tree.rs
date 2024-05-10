use crate::classification::alias::FType;
use crate::classification::mondrian_node::{Node, Stats};

use half::vec;
use ndarray::Array1;

use num::{Float, FromPrimitive};
use rand::prelude::*;
use rand_distr::{Distribution, Exp};

use std::collections::HashSet;

use std::fmt;

use std::usize;

#[derive(Clone)]
pub struct MondrianTreeClassifier<F: FType> {
    n_features: usize,
    n_labels: usize,
    rng: ThreadRng,
    nodes: Vec<Node<F>>,
    root: Option<usize>,
}

impl<F: FType + fmt::Display> fmt::Display for MondrianTreeClassifier<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n┌ MondrianTreeClassifier")?;
        self.recursive_repr(self.root, f, "│ ")
    }
}
impl<F: FType + fmt::Display> MondrianTreeClassifier<F> {
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

impl<F: FType> MondrianTreeClassifier<F> {
    pub fn new(n_features: usize, n_labels: usize) -> Self {
        MondrianTreeClassifier::<F> {
            n_features,
            n_labels,
            rng: rand::thread_rng(),
            nodes: vec![],
            root: None,
        }
    }

    fn create_leaf(&mut self, x: &Array1<F>, y: usize, parent: Option<usize>, time: F) -> usize {
        let mut node = Node::<F> {
            parent,
            time, // F::from(1e9).unwrap(), // Very large value
            is_leaf: true,
            min_list: x.clone(),
            max_list: x.clone(),
            feature: 0,
            threshold: F::zero(),
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
        time: F,
        exp_sample: F,
        node_idx: usize,
        y: usize,
        extensions_sum: F,
    ) -> F {
        if self.nodes[node_idx].is_dirac(y) {
            // println!(
            //     "go_downwards() - node: {node_idx} - extensions_sum: {:?} - all same class",
            //     extensions_sum
            // );
            return F::zero();
        }

        if extensions_sum > F::zero() {
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

        F::zero()
    }

    fn go_downwards(&mut self, node_idx: usize, x: &Array1<F>, y: usize) -> usize {
        let time = self.nodes[node_idx].time;
        let node_min_list = &self.nodes[node_idx].min_list;
        let node_max_list = &self.nodes[node_idx].max_list;
        let extensions = {
            let e_min = (node_min_list - x).mapv(|v| F::max(v, F::zero()));
            let e_max = (x - node_max_list).mapv(|v| F::max(v, F::zero()));
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
        let split_time = self.compute_split_time(time, exp_sample, node_idx, y, extensions.sum());
        if split_time > F::zero() {
            // Here split the current node: if leaf we add children, otherwise
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

            let (lower_bound, upper_bound) = if x[feature] > node_min_list[feature] {
                (
                    node_min_list[feature].to_f32().unwrap(),
                    x[feature].to_f32().unwrap(),
                )
            } else {
                (
                    x[feature].to_f32().unwrap(),
                    node_max_list[feature].to_f32().unwrap(),
                )
            };
            let threshold = F::from_f32(self.rng.gen_range(lower_bound..upper_bound)).unwrap();
            // DEBUG: split in the middle
            let threshold = F::from_f32((lower_bound + upper_bound) / 2.0).unwrap();

            let mut min_list = node_min_list.clone();
            let mut max_list = node_max_list.clone();
            min_list.zip_mut_with(x, |a, &b| *a = F::min(*a, b));
            max_list.zip_mut_with(x, |a, &b| *a = F::max(*a, b));

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
            node.min_list.zip_mut_with(x, |a, b| *a = F::min(*a, *b));
            node.max_list.zip_mut_with(x, |a, b| *a = F::max(*a, *b));
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
    pub fn partial_fit(&mut self, x: &Array1<F>, y: usize) {
        self.root = match self.root {
            None => Some(self.create_leaf(x, y, None, F::zero())),
            Some(root_idx) => Some(self.go_downwards(root_idx, x, y)),
        };
        // println!("partial_fit() tree post {}", self);
    }

    fn fit(&self) {
        unimplemented!("Make the program first work with 'partial_fit', then implement this")
    }

    /// Note: In Nel215 codebase should work on multiple records, here it's
    /// working only on one, so it's the same as "predict()".
    pub fn predict_proba(&self, x: &Array1<F>) -> Array1<F> {
        // println!("predict_proba() - tree size: {}", self.nodes.len());
        // self.test_tree();
        self.predict(x, self.root.unwrap(), F::one())
    }

    fn predict(&self, x: &Array1<F>, node_idx: usize, p_not_separated_yet: F) -> Array1<F> {
        let node = &self.nodes[node_idx];

        // Probability 'p' of the box not splitting.
        //     eta (box dist): larger distance, more prob of splitting
        //     d (time delta with parent): more dist with parent, more prob of splitting
        let p = {
            let d = node.time - self.get_parent_time(node_idx);
            let dist_max = (x - &node.max_list).mapv(|v| F::max(v, F::zero()));
            let dist_min = (&node.min_list - x).mapv(|v| F::max(v, F::zero()));
            let eta = dist_min.sum() + dist_max.sum();
            F::one() - (-d * eta).exp()
        };

        // Generate a result for the current node using its statistics.
        let res = node.stats.create_result(x, p_not_separated_yet * p);

        let w = p_not_separated_yet * (F::one() - p);
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

    fn get_parent_time(&self, node_idx: usize) -> F {
        // If node is root, time is 0
        match self.nodes[node_idx].parent {
            Some(parent_idx) => self.nodes[parent_idx].time,
            None => F::from_f32(0.0).unwrap(),
        }
    }

    pub fn get_tree_size(&self) -> usize {
        self.nodes.len()
    }

    /////////////////////////////////////////////////////////////
    /// Caching
    /////////////////////////////////////////////////////////////

    // Get node with highest count which has still not been visited.
    fn get_node_highest_chance(
        &self,
        nodes: &Vec<(bool, &Node<F>)>,
        nodes_chances: &Vec<(usize, usize)>,
    ) -> usize {
        for (node_idx, _) in nodes_chances {
            let (is_visited, _) = nodes[*node_idx];
            if !is_visited {
                return *node_idx;
            }
        }
        panic!("Should not get here");
    }

    /// Given a starting node 'node_idx' find path with highest count.
    /// e.g. node_idx=0 -> [0, 1, 3]
    /// e.g. node_idx=2 -> [2, 5]
    ///
    /// ┌─ Node 0: count=10
    /// │  ├─ Node 1: count=7
    /// │  │  ├─ Node 3: count=5
    /// │  │  └─ Node 4: count=2
    /// │  └─ Node 2: count=3
    /// │     ├─ Node 5: count=2
    /// │     └─ Node 6: count=1
    fn get_node_chain(&self, node_idx: usize, nodes: &Vec<(bool, &Node<F>)>) -> Vec<usize> {
        let mut node_chain = vec![];
        let (_, mut node) = nodes[node_idx];
        node_chain.push(node_idx);
        loop {
            if node.is_leaf {
                return node_chain;
            }
            let left_idx = node.left.unwrap();
            let right_idx = node.right.unwrap();
            let count_l = self.nodes[left_idx].stats.counts.sum();
            let count_r = self.nodes[right_idx].stats.counts.sum();
            let node_idx = if count_l >= count_r {
                left_idx
            } else {
                right_idx
            };
            (_, node) = nodes[node_idx];
            node_chain.push(node_idx);
        }
    }

    /// Get ordered nodes according to [1].
    /// [1] Chen et al. (2022). Efficient Realization of Decision Trees for Real-Time Inference.
    ///     ACM Transactions on Embedded Computing Systems, 21(6), 1–26. https://doi.org/10.1145/3508019
    ///
    /// Returns e.g. [0, 1, 3, 2, 5, 4, 6]
    ///
    /// ┌─ Node 0: count=10
    /// │  ├─ Node 1: count=7
    /// │  │  ├─ Node 3: count=5
    /// │  │  └─ Node 4: count=2
    /// │  └─ Node 2: count=3
    /// │     ├─ Node 5: count=2
    /// │     └─ Node 6: count=1
    fn get_new_index_order(&self) -> Vec<usize> {
        // nodes: [(is_visited, Node)]
        // e.g. [(false, node1), (true, node2), ...]
        let mut nodes: Vec<(bool, &Node<F>)> = self.nodes.iter().map(|n| (false, n)).collect();

        // nodes_chances reversed: [(node_idx, node_count)]
        // e.g. [(0, 10), (2, 8), (1, 7), ...]
        let mut nodes_chances: Vec<(usize, usize)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.stats.counts.sum()))
            .collect();
        nodes_chances.sort_by_key(|k| k.1);
        nodes_chances.reverse();

        let mut ordered_indicies = vec![];
        while nodes.len() != ordered_indicies.len() {
            let node_idx = self.get_node_highest_chance(&nodes, &nodes_chances);
            let node_chain = self.get_node_chain(node_idx, &nodes);
            ordered_indicies.extend(&node_chain);
            for i in node_chain {
                nodes[i].0 = true;
            }
            assert!(ordered_indicies.len() <= nodes.len());
        }
        ordered_indicies
    }

    /// Nodes in the vector are sorted by the likelihood of access. By arranging nodes
    /// so that the most probable next node is positioned next in the vector, spatial
    /// locality is enhanced, improving cache efficiency.
    ///
    /// Follows implementation by [1].
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
        // println!("cache_sort() - tree pre {}", self);

        let new_order: Vec<usize> = self.get_new_index_order();

        // e.g. new_order=[2, 0, 1] -> [2]=0, [0]=1, [1]=2 -> new_order_mapped=[1, 2, 0]
        let mut new_order_mapped = vec![0; new_order.len()];
        for (index, &value) in new_order.iter().enumerate() {
            new_order_mapped[value] = index;
        }

        let mut nodes_reordered: Vec<Node<F>> = Vec::with_capacity(self.nodes.len());
        for i in &new_order {
            let mut new_node = self.nodes[*i].clone();
            new_node.right = new_node.right.map(|v| new_order_mapped[v]);
            new_node.left = new_node.left.map(|v| new_order_mapped[v]);
            new_node.parent = new_node.parent.map(|v| new_order_mapped[v]);
            nodes_reordered.push(new_node);
        }

        self.nodes = nodes_reordered;

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
