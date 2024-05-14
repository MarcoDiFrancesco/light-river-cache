use light_river::classification::mondrian_forest::MondrianForestClassifier;

use light_river::common::ClassifierTarget;
use light_river::datasets::synthetic::Synthetic;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use num::ToPrimitive;

use std::fs::File;
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["feature_1", "feature_2", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    let sample = transactions.into_iter().next();
    let observation = sample.unwrap().unwrap().get_observation();
    let mut out: Vec<String> = observation.iter().map(|(k, _)| k.clone()).collect();
    out.sort();
    out
}

/// Get list of labels.
///
/// e.g. ["0", "1", "2"]
fn get_labels(transactions: IterCsv<f32, File>) -> Vec<String> {
    let mut labels = vec![];
    for t in transactions {
        let data = t.unwrap();
        // TODO: use instead 'to_classifier_target' and a vector of 'ClassifierTarget'
        let target = data.get_y().unwrap()["label"].to_string();
        if !labels.contains(&target) {
            labels.push(target);
        }
    }
    labels
}

fn get_dataset_size(transactions: IterCsv<f32, File>) -> usize {
    let mut length = 0;
    for _ in transactions {
        length += 1;
    }
    length
}

fn train_forest(
    mf: &mut MondrianForestClassifier,
    features: &Vec<String>,
    labels: &Vec<String>,
    dataset_size: usize,
) {
    let mut score_total = 0.0;
    let transactions = Synthetic::load_data();

    const CACHE_SORT: bool = true;
    const CACHE_FREQ: usize = 25_000;
    if CACHE_SORT {
        println!("Cache sort. Sorting every {} iterations.", CACHE_FREQ);
    } else {
        println!("No cache sort");
    }

    let train_instant = Instant::now();
    let mut cache_time = 0.0;
    for (idx, transaction) in transactions.enumerate() {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let x = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        let y = data.to_classifier_target("label").unwrap();
        let y = match y {
            ClassifierTarget::String(y) => y,
            _ => unimplemented!(),
        };
        let y = labels.clone().iter().position(|l| l == &y).unwrap();

        // println!("=M=1 x {}", x);

        // Skip first sample since tree has still no node
        if idx != 0 {
            let score = mf.score(&x, y);
            score_total += score;
            // println!(
            //     "Accuracy: {} / {} = {}",
            //     score_total,
            //     dataset_size - 1,
            //     score_total / idx.to_f32().unwrap()
            // );
        }

        // println!("=M=1 partial_fit {x}");
        mf.partial_fit(&x, y);

        if CACHE_SORT & (idx % CACHE_FREQ == 0) {
            let cache_instant = Instant::now();
            mf.cache_sort();
            cache_time += cache_instant.elapsed().as_micros().to_f32().unwrap() / 1000f32;
            // println!("Sorted at inedex: {}", idx);
        }
    }
    let train_time = train_instant.elapsed().as_micros().to_f32().unwrap() / 1000f32;
    let discounted_time = train_time - cache_time;

    println!("Discounted time (total-sorting): {}ms", discounted_time);

    // Accuracy does not include first sample.
    println!(
        "Accuracy: {} / {} = {}",
        score_total,
        dataset_size - 1,
        score_total / (dataset_size - 1).to_f32().unwrap()
    );
    let forest_size = mf.get_forest_size();
    println!("Forest tree sizes: {:?}", forest_size);
}

fn main() {
    let n_trees: usize = 1;

    let transactions_f = Synthetic::load_data();
    let features = get_features(transactions_f);

    let transactions_c = Synthetic::load_data();
    let labels = get_labels(transactions_c);
    println!("labels: {labels:?}, features: {features:?}");
    let mut mf: MondrianForestClassifier =
        MondrianForestClassifier::new(n_trees, features.len(), labels.len());

    let transactions_l = Synthetic::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    train_forest(&mut mf, &features, &labels, dataset_size);
}
