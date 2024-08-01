use csv::WriterBuilder;
use light_river::mondrian_forest::mondrian_forest::MondrianForestRegressor;

use light_river::common::{RegTarget, Regressor};
use light_river::datasets::machine_degradation::MachineDegradation;
use light_river::stream::iter_csv::IterCsv;
use ndarray::Array1;
use num::ToPrimitive;

use std::fs::{File, OpenOptions};
use std::time::Instant;

/// Get list of features of the dataset.
///
/// e.g. features: ["H.e", "UD.t.i", "H.i", ...]
fn get_features(transactions: IterCsv<f32, File>) -> Vec<String> {
    let sample = transactions.into_iter().next();
    let observation = sample.unwrap().unwrap().get_observation();
    let mut out: Vec<String> = observation.iter().map(|(k, _)| k.clone()).collect();
    out.sort();
    out
}

fn get_dataset_size(transactions: IterCsv<f32, File>) -> usize {
    let mut length = 0;
    for _ in transactions {
        length += 1;
    }
    length
}

fn train_forest(
    mf: &mut MondrianForestRegressor<f32>,
    features: &Vec<String>,
    dataset_size: usize,
) {
    let mut err_total = 0.0;
    let transactions = MachineDegradation::load_data();

    const CACHE_SORT: bool = false;
    const CACHE_FREQ: usize = 250_000 - 2;
    if CACHE_SORT {
        println!("Cache sort. Sorting every {} iterations.", CACHE_FREQ);
    } else {
        println!("No cache sort");
    }

    let task = "reg";
    let is_opt = if CACHE_SORT { "opt" } else { "notopt" };

    let path_train_times = format!("res_{task}_{is_opt}_times.csv");
    std::fs::File::create(&path_train_times).unwrap();
    let mut wtr_train_times = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open(&path_train_times)
                .unwrap(),
        );

    let path_tree_size = format!("res_{task}_{is_opt}_tree_size.csv");
    std::fs::File::create(&path_tree_size).unwrap();
    let mut wtr_tree_size = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open(&path_tree_size)
                .unwrap(),
        );

    let path_depth = format!("res_{task}_{is_opt}_depth.csv");
    std::fs::File::create(&path_depth).unwrap();
    let mut wtr_depth = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(OpenOptions::new().append(true).open(&path_depth).unwrap());

    let path_sorted_count = format!("res_{task}_{is_opt}_sorted_count.csv");
    std::fs::File::create(&path_sorted_count).unwrap();
    let mut wtr_sorted_count = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open(&path_sorted_count)
                .unwrap(),
        );

    let path_sort_time = format!("res_{task}_{is_opt}_sort_time.csv");
    std::fs::File::create(&path_sort_time).unwrap();
    let mut wtr_sort_time = WriterBuilder::new()
        .quote_style(csv::QuoteStyle::Never)
        .from_writer(
            OpenOptions::new()
                .append(true)
                .open(&path_sort_time)
                .unwrap(),
        );

    // Total train time
    let mut train_time_tot = 0.0;
    // Cumulative time for 1.000 iters train+score, reset on cache sort
    let mut score_fit_time_cum = 0;

    for (idx, transaction) in transactions.enumerate() {
        let data = transaction.unwrap();

        let x = data.get_observation();
        let x = Array1::<f32>::from_vec(features.iter().map(|k| x[k]).collect());

        let y = data.to_regression_target("pCut::Motor_Torque").unwrap();

        // println!("=M=1 x:{}, idx: {}", x, idx);

        let mut train_time_str = String::new();
        let score_instant: Instant = Instant::now();

        // Skip first sample since tree has still no node
        if idx != 0 {
            let pred = mf.predict_one(&x, &y);
            let err = (pred - y).powi(2);
            err_total += err;
            // println!("pred: {pred}, y: {y}, err: {err}");
        }
        let score_time = score_instant.elapsed().as_nanos();

        // println!("=M=1 partial_fit {x}");
        let fit_instant = Instant::now();
        mf.learn_one(&x, &y);
        let fit_time = fit_instant.elapsed().as_nanos();

        train_time_str
            .push_str(format!("{},{},{}", score_time, fit_time, score_time + fit_time).as_str());
        train_time_tot += score_instant.elapsed().as_micros().to_f32().unwrap() / 1_000f32;

        score_fit_time_cum += score_time + fit_time;

        if (idx % CACHE_FREQ == 0) {
            let mut cache_time = 0;
            if CACHE_SORT {
                let cache_instant = Instant::now();
                mf.cache_sort();
                cache_time = cache_instant.elapsed().as_nanos();
            }

            // Mesure number of nodes in the tree
            wtr_tree_size
                .write_record(&[mf.get_forest_size().to_string()])
                .unwrap();
            wtr_tree_size.flush().unwrap();

            // Mesure tree depths metrics: avg, max, etc.
            let (node_n, optim, avg, avg_w, max) = mf.get_forest_depth();
            let depth_str = format!("{},{},{},{},{}", node_n, optim, avg, avg_w, max);
            wtr_depth.write_record(&[depth_str]).unwrap();
            wtr_depth.flush().unwrap();

            // Measure: Sorted nodes vs Unsorted nodes
            let (sorted_count, unsorted_count) = mf.get_sorted_count();
            let sorted_count_str = format!("{},{}", sorted_count, unsorted_count);
            wtr_sorted_count.write_record(&[sorted_count_str]).unwrap();
            wtr_sorted_count.flush().unwrap();

            // Measure: Time taken to sort vs Time saved by sorting
            let node_n = mf.get_forest_size();
            let sort_time_str = format!("{},{},{},{}", node_n, idx, cache_time, score_fit_time_cum);
            wtr_sort_time.write_record(&[sort_time_str]).unwrap();
            wtr_sort_time.flush().unwrap();

            score_fit_time_cum = 0;
        }
        wtr_train_times.write_record(&[train_time_str]).unwrap();
        wtr_train_times.flush().unwrap();
    }

    println!("Score+fit time (excuding cache sort): {}ms", train_time_tot);

    // Accuracy does not include first sample.
    println!(
        "Accuracy: {} / {} = {}",
        err_total,
        dataset_size - 1,
        err_total / (dataset_size - 1).to_f32().unwrap()
    );
    let forest_size = mf.get_forest_size();
    println!("Forest tree sizes: {:?}", forest_size);
}

fn main() {
    let n_trees: usize = 1;

    let transactions_f = MachineDegradation::load_data();
    let features = get_features(transactions_f);

    println!("features: {features:?}");

    let mut mf: MondrianForestRegressor<f32> =
        MondrianForestRegressor::new(n_trees, features.len());

    let transactions_l = MachineDegradation::load_data();
    let dataset_size = get_dataset_size(transactions_l);

    train_forest(&mut mf, &features, dataset_size);
}
