def load_amem_dataset(dataset_dir, dataset_type="keywords_tags"):
    """
    Load our custom AMEM dataset (keywords_tags or content)
    """
    from datasets import Dataset, DatasetDict
    import os
    
    # Define file paths
    train_path = os.path.join(dataset_dir, f"analyze_content_{dataset_type}.train.tokens")
    valid_path = os.path.join(dataset_dir, f"analyze_content_{dataset_type}.valid.tokens")
    test_path = os.path.join(dataset_dir, f"analyze_content_{dataset_type}.test.tokens")
    
    # Read files and create datasets
    def read_tokens_file(file_path):
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by "= = Sample" to get individual samples
        samples = content.split("= = Sample")
        samples = [sample.strip() for sample in samples if sample.strip()]
        
        return [{"text": sample} for sample in samples]
    
    train_data = read_tokens_file(train_path)
    valid_data = read_tokens_file(valid_path)
    test_data = read_tokens_file(test_path)
    
    # Create datasets using DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(valid_data),
        'test': Dataset.from_list(test_data)
    })
    
    return dataset

def load_mobile_agent_v3_vl_dataset(data_path, train_split=0.8, val_split=0.1, test_split=0.1):
    """
    Load Mobile Agent v3 VL dataset from trajectory data directory.
    Converts the data to HuggingFace Dataset format for fine-tuning.
    
    Args:
        data_path: Path to the trajectory data directory (e.g., traj_data_save_2025-12-04_23-02-43)
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        test_split: Fraction of data for testing (default: 0.1)
    
    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits
    """
    from datasets import Dataset, DatasetDict
    from pathlib import Path
    import json
    import random
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # Find all task directories
    task_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Found {len(task_dirs)} task directories")
    
    # Collect supervised (prompt -> output) pairs from all three JSON files and image paths
    all_samples = []
    
    for task_dir in task_dirs:
        planning_file = task_dir / "planning.json"
        action_data_file = task_dir / "action_data.json"
        action_reflect_file = task_dir / "action_reflect.json"
        
        # Load all three JSON files
        planning_data = []
        action_data = []
        action_reflect_data = []
        
        if planning_file.exists():
            try:
                with open(planning_file, 'r', encoding='utf-8') as f:
                    planning_data = json.load(f)
            except Exception as e:
                print(f"Error loading planning.json from {task_dir}: {e}")
        
        if action_data_file.exists():
            try:
                with open(action_data_file, 'r', encoding='utf-8') as f:
                    action_data = json.load(f)
            except Exception as e:
                print(f"Error loading action_data.json from {task_dir}: {e}")
        
        if action_reflect_file.exists():
            try:
                with open(action_reflect_file, 'r', encoding='utf-8') as f:
                    action_reflect_data = json.load(f)
            except Exception as e:
                print(f"Error loading action_reflect.json from {task_dir}: {e}")
        
        # Collect all step indices from all three files
        all_step_indices = set()
        for step in planning_data:
            all_step_indices.add(step.get('step_idx', 0))
        for step in action_data:
            all_step_indices.add(step.get('step_idx', 0))
        for step in action_reflect_data:
            all_step_indices.add(step.get('step_idx', 0))
        
        # Process each step index
        for step_idx in sorted(all_step_indices):
            # Find corresponding screenshot (same for all JSON files with same step_idx)
            screenshot_path = task_dir / f"screenshot_{step_idx}.png"
            if not screenshot_path.exists():
                # Try to find any screenshot if exact match not found
                screenshots = sorted(task_dir.glob("screenshot_*.png"))
                if screenshots:
                    screenshot_path = screenshots[0]  # Use first available
                else:
                    screenshot_path = None
            
            screenshot_path_str = str(screenshot_path) if screenshot_path else None
            
        # Collect (prompt -> output) from planning.json
        for step in planning_data:
            if step.get("step_idx", 0) == step_idx:
                prompt = step.get("prompt_planning", "") or ""
                output = step.get("output_planning", "") or ""
                if prompt.strip() and output.strip():
                    all_samples.append(
                        {
                            "prompt": prompt,
                            "response": output,
                            "task": "planning",
                            "image_path": screenshot_path_str,
                        }
                    )
                break
            
        # Collect (prompt -> output) from action_data.json
        for step in action_data:
            if step.get("step_idx", 0) == step_idx:
                prompt = step.get("prompt_action", "") or ""
                output = step.get("output_action", "") or ""
                if prompt.strip() and output.strip():
                    all_samples.append(
                        {
                            "prompt": prompt,
                            "response": output,
                            "task": "action",
                            "image_path": screenshot_path_str,
                        }
                    )
                break
            
        # Collect (prompt -> output) from action_reflect.json
        for step in action_reflect_data:
            if step.get("step_idx", 0) == step_idx:
                prompt = step.get("prompt_action_reflect", "") or ""
                output = step.get("output_action_reflect", "") or ""
                if prompt.strip() and output.strip():
                    all_samples.append(
                        {
                            "prompt": prompt,
                            "response": output,
                            "task": "action_reflect",
                            "image_path": screenshot_path_str,
                        }
                    )
                break
    
    if len(all_samples) == 0:
        raise ValueError("No valid data found in the specified path")
    
    print(
        f"Collected {len(all_samples)} supervised samples (prompt->response) from planning.json, action_data.json, and action_reflect.json"
    )
    
    # Shuffle samples
    random.shuffle(all_samples)
    
    # Split dataset
    total_samples = len(all_samples)
    train_end = int(total_samples * train_split)
    val_end = train_end + int(total_samples * val_split)
    
    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]
    
    # Create datasets using DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    print(f"Dataset splits - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    return dataset

def preprocess_dataset(dataset, tokenizer, max_length=512):
    """
    Filters out empty strings in the text column and tokenizes the dataset.
    """
    # Remove empty strings
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    
    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns="text")
    
    return tokenized_dataset
