#!/bin/bash
for i in {1..30}; do 
  echo "RUN: ${i}" >> run_synthetic_output.txt
  RUSTFLAGS=-Awarnings cargo run --release --example synthetic >> run_synthetic_output.txt
done
