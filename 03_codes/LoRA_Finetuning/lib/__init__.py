import os
import importlib

# The list of .py files to import functions from (without the .py extension)
files_to_import = ["dataset", "LoRA_custom_mapping_func", "LoRA_custom_layer", "LoRA_custom_vptq", "LoRA_custom_variants", "LoRA_custom_config", "LoRA_custom_dora", "custom_nvfp", "custom_nvfp_for_saves", "precision"]

# Optional modules that may not be available (e.g., require bitsandbytes)
optional_modules = ["LoRA_custom_nvfp_for_saves", "LoRA_custom_nvfp", "LoRA_custom_bnb"]

# Iterate over each specified file
for module_name in files_to_import:
    # Import the module dynamically
    module = importlib.import_module(f".{module_name}", package=__name__)
    
    # Import all functions from the module into the current namespace
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # Check if the attribute is a function and not private (doesn't start with _)
        if callable(attribute) and not attribute_name.startswith("_"):
            globals()[attribute_name] = attribute

# Import optional modules (skip if import fails)
for module_name in optional_modules:
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        # Import all functions from the module into the current namespace
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            # Check if the attribute is a function and not private (doesn't start with _)
            if callable(attribute) and not attribute_name.startswith("_"):
                globals()[attribute_name] = attribute
    except (ImportError, ModuleNotFoundError) as e:
        # Silently skip optional modules that can't be imported
        pass