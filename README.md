# CS5242-Project: Multimodal Sentiment Analysis with CLIP

This project implements a multimodal sentiment classification model using features extracted from the CLIP (Contrastive Language–Image Pre-training) model. It allows for flexible configuration, experimentation with different model architectures (including single-modality baselines), and handles data preprocessing efficiently.

## Features

* **CLIP-Based:** Leverages powerful pre-trained image and text embeddings from CLIP.
* **Modular Structure:** Code organized into `src/` with separate modules for configuration, data handling, model definition, training, and evaluation.
* **Efficient Preprocessing:** Includes an offline preprocessing step (`src/preprocess.py`) to extract and save CLIP features, speeding up training significantly.
* **Configurable Model:**
    * Supports multimodal, image-only, and text-only modes.
    * Optional cross-attention fusion mechanism.
    * Optional linear projection layers after CLIP features.
    * Configurable MLP classifier head structure.
    * Option to freeze or fine-tune CLIP weights.
* **Experimentation Framework:**
    * Uses `run_experiment.py` as the main entry point.
    * Configuration managed via `src/config.py` and overridable via command-line arguments.
    * Outputs (logs, best model, config, results) saved to unique directories under `output/` for easy comparison.
* **Regularization:** Includes Dropout and Weight Decay options, plus Early Stopping to mitigate overfitting.

## Usage

1.  **Data Preprocessing:**
    * This step extracts features using the CLIP processor and saves them to `preprocessed_data/`. It only needs to be run once unless your raw data or preprocessing configuration changes.
    * The main script `run_experiment.py` will **automatically** trigger this if the `preprocessed_data/train` directory is empty or if the `--force_preprocess` command-line argument is used.
    * To run it manually:
        ```bash
        python src/preprocess.py
        ```

2.  **Running Experiments:**
    * Use the `run_experiment.py` script to start training and evaluation.
    * Configuration is controlled by `src/config.py` but can be overridden with command-line arguments.

    * **Example: Run with default settings:**
        ```bash
        python run_experiment.py --experiment_name "baseline_run"
        ```
        This will save results to `output/baseline_run/`.

    * **Example: Run an image-only experiment with different batch size:**
        ```bash
        python run_experiment.py --experiment_name "image_only_bs64" --modality image --batch_size 64
        ```
        Results will be in `output/image_only_bs64/`.

    * **Example: Fine-tune CLIP with specific LRs and more dropout:**
        ```bash
        python run_experiment.py --experiment_name "finetune_clip_more_dropout" \
          --freeze_clip false \
          --learning_rate_clip 5e-7 \
          --learning_rate_head 5e-5 \
          --dropout_mlp 0.5
        ```

    * **Example: Force preprocessing before running:**
        ```bash
        python run_experiment.py --experiment_name "rerun_preprocess" --force_preprocess
        ```

    * Refer to `run_experiment.py` (`parser.add_argument` lines) and `src/config.py` for all available command-line arguments and their default values.

3.  **Output:**
    * All outputs for a specific run are saved in `output/<experiment_name>/`.
    * `config.json`: The exact configuration used for the run.
    * `logs.log`: Detailed logs from the run (training progress, validation scores, errors).
    * `best_model.pth`: The state dictionary of the model with the best validation accuracy.
    * `test_results.json`: Final evaluation metrics on the test set.

## TODO
1. multimodal + cross-attention + projection + CNN + linear projection layers + default MLP
2. multimodal + cross-attention + projection + (CNN) + linear projection layers + default MLP
3. multimodal + cross-attention + (projection) + CNN + linear projection layers + default MLP
4. multimodal + (cross-attention) + projection + CNN + linear projection layers + default MLP
5. multimodal + cross-attention + projection + CNN + linear projection layers + less MLP
6. multimodal + cross-attention + projection + CNN + linear projection layers + more MLP
7. image-only + CNN + linear projection layers + default MLP
8. image-only + (CNN) + linear projection layers + default MLP (optional)
9. text-only + CNN + linear projection layers + default MLP
10. text-only + (CNN) + linear projection layers + default MLP (optional)