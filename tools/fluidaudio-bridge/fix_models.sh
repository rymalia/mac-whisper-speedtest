#!/bin/bash
# FluidAudio Model Copy Script
#
# This script fixes FluidAudio's model loading issue by manually copying
# models from HuggingFace cache to Application Support.
#
# Background: FluidAudio downloads models to ~/.cache/huggingface/hub/ but
# fails to properly copy them to ~/Library/Application Support/FluidAudio/Models/
# This script completes the copy process manually.

set -e

echo "FluidAudio Model Fix Script"
echo "============================"
echo ""

# Define paths
HF_CACHE="$HOME/.cache/huggingface/hub"
APP_SUPPORT="$HOME/Library/Application Support/FluidAudio/Models"

# Check if HuggingFace cache exists
if [ ! -d "$HF_CACHE" ]; then
    echo "Error: HuggingFace cache not found at $HF_CACHE"
    echo "Please ensure models have been downloaded first."
    exit 1
fi

# Function to copy models for a specific version
copy_models() {
    local version=$1
    local hf_model_name=$2
    local target_dir_name=$3

    echo "Processing $version models..."

    # Find the snapshot directory
    local snapshot_dir=$(find "$HF_CACHE/models--FluidInference--$hf_model_name/snapshots" -type d -maxdepth 1 -mindepth 1 2>/dev/null | head -1)

    if [ -z "$snapshot_dir" ]; then
        echo "  ⚠️  $version models not found in HuggingFace cache"
        echo "      Expected: $HF_CACHE/models--FluidInference--$hf_model_name"
        return 1
    fi

    echo "  ✓ Found models in HuggingFace cache"

    # Check for required files
    local required_files=("Encoder.mlmodelc" "Decoder.mlmodelc" "Preprocessor.mlmodelc" "JointDecision.mlmodelc" "parakeet_vocab.json" "config.json")
    local missing_files=()

    for file in "${required_files[@]}"; do
        if [ ! -e "$snapshot_dir/$file" ]; then
            missing_files+=("$file")
        fi
    done

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo "  ⚠️  Incomplete models in cache. Missing files:"
        for file in "${missing_files[@]}"; do
            echo "      - $file"
        done
        return 1
    fi

    echo "  ✓ All required files found in cache"

    # Create target directory
    local target_dir="$APP_SUPPORT/$target_dir_name"
    mkdir -p "$target_dir"

    # Copy all required files
    echo "  → Copying models to Application Support..."
    for file in "${required_files[@]}"; do
        if [ -e "$target_dir/$file" ]; then
            rm -rf "$target_dir/$file"
        fi
        cp -r "$snapshot_dir/$file" "$target_dir/"
    done

    # Verify sizes
    local encoder_size=$(du -sh "$target_dir/Encoder.mlmodelc" | cut -f1)
    local decoder_size=$(du -sh "$target_dir/Decoder.mlmodelc" | cut -f1)

    echo "  ✓ Models copied successfully"
    echo "    - Encoder: $encoder_size"
    echo "    - Decoder: $decoder_size"
    echo "    - Preprocessor, JointDecision, vocab, config: ✓"
    echo ""

    return 0
}

# Copy v3 models (default for FluidAudio 0.8.0+)
echo "Checking v3 models (default for FluidAudio 0.8.0+)..."
if copy_models "v3" "parakeet-tdt-0.6b-v3-coreml" "parakeet-tdt-0.6b-v3-coreml"; then
    V3_SUCCESS=true
else
    V3_SUCCESS=false
fi

# Copy v2 models (fallback for FluidAudio 0.1.0)
echo "Checking v2 models (for FluidAudio 0.1.0)..."
if copy_models "v2" "parakeet-tdt-0.6b-v2-coreml" "parakeet-tdt-0.6b-v2-coreml"; then
    V2_SUCCESS=true
else
    V2_SUCCESS=false
fi

# Summary
echo "============================"
echo "Summary:"
if [ "$V3_SUCCESS" = true ]; then
    echo "  ✓ v3 models ready (FluidAudio 0.8.0+)"
fi
if [ "$V2_SUCCESS" = true ]; then
    echo "  ✓ v2 models ready (FluidAudio 0.1.0)"
fi

if [ "$V3_SUCCESS" = false ] && [ "$V2_SUCCESS" = false ]; then
    echo "  ✗ No models could be copied"
    echo ""
    echo "Please download models first using:"
    echo "  cd tools/fluidaudio-bridge"
    echo "  swift build -c release"
    echo "  # Then run the bridge once to trigger model download"
    exit 1
fi

echo ""
echo "FluidAudio models are now ready!"
echo "You can test with:"
echo "  ./tools/fluidaudio-bridge/.build/release/fluidaudio-bridge <audio.wav> --format json"
