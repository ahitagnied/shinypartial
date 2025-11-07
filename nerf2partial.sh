#!/bin/bash

DIRECTION="-1,-1,1"
VERTICAL_SPAN=180
HORIZONTAL_SPAN=180
INPUT_DIR="refnerf"
OUTPUT_DIR="refnerf-partial"

mkdir -p "$OUTPUT_DIR"

for dataset in "$INPUT_DIR"/*/; do
    dataset_name=$(basename "$dataset")
    
    if [ ! -d "$dataset" ]; then
        continue
    fi
    
    if [ ! -f "$dataset/transforms_train.json" ] || [ ! -f "$dataset/transforms_test.json" ]; then
        echo "skipping $dataset_name - missing transforms files"
        continue
    fi
    
    echo "processing $dataset_name..."
    
    python nerf2partial.py "$dataset" "$DIRECTION" "$VERTICAL_SPAN" "$HORIZONTAL_SPAN" "$OUTPUT_DIR" "${dataset_name}-partial"
    
    if [ $? -eq 0 ]; then
        echo "successfully created subset for $dataset_name"
    else
        echo "failed to create subset for $dataset_name"
    fi
    
    echo "---"
done

echo "all datasets processed"
