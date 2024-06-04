use csv::{Writer, WriterBuilder};
use light_river::classification::mondrian_forest::MondrianForestClassifier;

use light_river::common::ClassifierTarget;
use light_river::datasets::synthetic::Synthetic;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use num::ToPrimitive;

use std::fmt::format;
use std::fs::{File, OpenOptions};
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
    mf: &mut MondrianForestClassifier<f32>,
    features: &Vec<String>,
    labels: &Vec<String>,
    dataset_size: usize,
) {
    let mut score_total = 0.0;
    let transactions = Synthetic::load_data();

    const CACHE_SORT: bool = false;
    const CACHE_FREQ: usize = 1_000;
    if CACHE_SORT {
        println!("Cache sort. Sorting every {} iterations.", CACHE_FREQ);
    } else {
        println!("No cache sort");
    }

    std::fs::File::create("run_synthetic_output_times.csv").unwrap();
    let mut wtr_train_times = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open("run_synthetic_output_times.csv")
                .unwrap(),
        );

    std::fs::File::create("run_synthetic_output_tree_size.csv").unwrap();
    let mut wtr_tree_size = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open("run_synthetic_output_tree_size.csv")
                .unwrap(),
        );

    std::fs::File::create("run_synthetic_output_depth.csv").unwrap();
    let mut wtr_depth = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open("run_synthetic_output_depth.csv")
                .unwrap(),
        );

    std::fs::File::create("run_synthetic_output_sorted_count.csv").unwrap();
    let mut wtr_sorted_count = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open("run_synthetic_output_sorted_count.csv")
                .unwrap(),
        );

    let mut train_time_tot = 0.0;
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

        let mut train_time_str = String::new();
        let score_instant: Instant = Instant::now();

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
        let score_time = score_instant.elapsed().as_nanos();

        // println!("=M=1 partial_fit {x}");
        let fit_instant = Instant::now();
        mf.partial_fit(&x, y);
        let fit_time = fit_instant.elapsed().as_nanos();

        train_time_str
            .push_str(format!("{},{},{}", score_time, fit_time, score_time + fit_time).as_str());
        train_time_tot += score_instant.elapsed().as_micros().to_f32().unwrap() / 1_000f32;

        if (idx % CACHE_FREQ == 0) {
            if CACHE_SORT {
                let cache_time = Instant::now();
                mf.cache_sort();
                // train_time_str.push_str(format!(" CACHING time: {}", cache_time.elapsed().as_nanos()).as_str());
                // println!("Sorted at inedex: {}", idx);
            }

            // Mesure tree size
            wtr_tree_size
                .write_record(&[mf.get_forest_size().to_string()])
                .unwrap();
            wtr_tree_size.flush().unwrap();

            // Mesure depths
            let (node_n, optim, avg, avg_w, max) = mf.get_forest_depth();
            let depth_str = format!("{},{},{},{},{}", node_n, optim, avg, avg_w, max);
            wtr_depth.write_record(&[depth_str]).unwrap();
            wtr_depth.flush().unwrap();

            // Count ordered nodes
            let (sorted_count, unsorted_count) = mf.get_sorted_count();
            let sorted_count_str = format!("{},{}", sorted_count, unsorted_count);
            wtr_sorted_count.write_record(&[sorted_count_str]).unwrap();
            wtr_sorted_count.flush().unwrap();
        }
        wtr_train_times.write_record(&[train_time_str]).unwrap();
        wtr_train_times.flush().unwrap();
    }

    println!("Score+fit time (excuding cache sort): {}ms", train_time_tot);

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
    let mut mf: MondrianForestClassifier<f32> =
        MondrianForestClassifier::new(n_trees, features.len(), labels.len());

    let transactions_l = Synthetic::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    train_forest(&mut mf, &features, &labels, dataset_size);
}
