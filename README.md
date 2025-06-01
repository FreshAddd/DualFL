# DualModelLinux - Dual-Model Fault Localization System Based on Graph Neural Networks

This project is the source code for the paper "DualFL: A Method-Statement Collaborative Fault Localization Approach Based on Dual-Layer Heterogeneous Graphs".

## Project Structure

- `model/` - Model implementation code
  - `dual_train_run.py` - Main entry file, containing training and evaluation processes
  - Other model files (CNN, Transformer, etc.)
- `dataset/` - Dataset loading and processing
  - `create_embedding_dict_v2.py` - Vector generation for nodes
  - `graph_embedding_merge.py` - Node attribute construction based on neighbor relationships
- `mypraser/` - Code parsing tools
- `defects4j/` - Defects4J project configuration and tools
- `data/` - Data file directory
  - `graph/` - Graph data files
    - `Lang/` - Lang project data
    - `Time/` - Time project data
    - `Mockito/` - Mockito project data
    - `Chart/` - Chart project data
    - `Math/` - Math project data

## Environment Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA support (for GPU acceleration)
- Other dependencies can be found in the code imports

## Installation Guide

1. Clone the project to local
2. Install required dependencies:
   ```
   pip install torch pytorch-lightning tensorboard tqdm numpy
   ```

## Usage Instructions

### Main Entry

The main entry file of the project is `model/dual_train_run.py`, run this file directly to start training and evaluation:

```bash
python model/dual_train_run.py
```

### Project Configuration

1. Project configuration is in the `get_projects()` function in `defects4j/utils.py`:
   - This function returns the list of projects to process, which can be modified as needed

2. Data file path configuration is in `config.py`:
   ```python
   self.path_dataset_home = os.path.join(current_dir, "data")
   ```
   - The default path is the `data` folder in the project root directory

### Data Description

- Complete data for Lang, Time, Mockito, Chart, and Math projects have been uploaded.

- Training and evaluation use PKL files in the following format:
  ```
  data/graph/[Project]/[Version]/[Project]_[Version]_graph_embedding_merge.pkl
  ```
  For example: `data/graph/Lang/1/Lang_1_graph_embedding_merge.pkl`

- Each PKL file contains four types of nodes:
  - Methods
  - Statements
  - Passed test cases
  - Failed test cases

- Each node contains both its own attributes and the attributes of neighboring nodes. All attributes have been processed using GraphCodeBERT and Node2Vec to form vectors that can be directly used for training.

- The Closure project data is not included due to its large file size. If needed, you can generate it using the models' data structure from the Defects4J dataset.

- To generate your own data, you can use:
  - `dataset/create_embedding_dict_v2.py` for vector generation
  - `dataset/graph_embedding_merge.py` for constructing node attributes based on neighbor relationships

### Output Files

- Training models are saved in the `DualFL_Model_New/` directory
- Evaluation results are saved in the `model_output_new/` directory

## Notes

1. Before running, ensure that data files are correctly placed in the configured path
2. The training process requires GPU support, please ensure the CUDA environment is correctly configured
3. By default, the training process will clean up intermediate model files to save space

## Modifying Test Versions

To modify the versions to be tested, edit the following in `dual_train_run.py`:

```python
active_bug = [1, 3]  # Modify to the desired version numbers
``` 