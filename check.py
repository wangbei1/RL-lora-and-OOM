import torch

def inspect_checkpoint(file_path, top_key):
    print("\n" + "="*80)
    print(f"File: {file_path}")
    print(f"Target Key: {top_key}")
    print("="*80)

    try:
        # Load data to CPU
        data = torch.load(file_path, map_location="cpu")
        
        # Access the internal model object
        model_obj = data.get(top_key, None)
        if model_obj is None:
            print(f"Error: Key '{top_key}' not found. Available top-level keys: {list(data.keys())}")
            return

        # Extract state_dict keys
        if isinstance(model_obj, dict):
            all_keys = list(model_obj.keys())
        elif hasattr(model_obj, 'state_dict'):
            all_keys = list(model_obj.state_dict().keys())
        else:
            print(f"Warning: Object is type {type(model_obj)}. Listing attributes instead.")
            all_keys = dir(model_obj)

        print(f"Total internal keys found: {len(all_keys)}")

        # Print first 50 keys to understand the architecture path
        print("\n--- Sample Internal Keys (First 50) ---")
        for k in all_keys[:50]:
            print(f"  {k}")

        # Extract unique module names (the part before .weight or .bias)
        # e.g., 'blocks.0.self_attn.qkv_proj.weight' -> 'qkv_proj'
        potential_modules = set()
        for k in all_keys:
            if '.weight' in k:
                parts = k.split('.')
                if len(parts) >= 2:
                    potential_modules.add(parts[-2])

        print("\n" + "*"*60)
        print("SUGGESTED TARGET_MODULES FOR LORA:")
        print("*"*60)
        
        # Filter for layers typically used in LoRA (Attention/MLP projections)
        keywords = ['proj', 'qkv', 'attn', 'query', 'key', 'value', 'to_']
        refined_targets = [m for m in potential_modules if any(kw in m.lower() for kw in keywords)]
        
        if refined_targets:
            print(sorted(refined_targets))
        else:
            print("No projection layers detected automatically.")
            print("Please check the 'Sample Internal Keys' list above manually.")
        print("*"*60)

    except Exception as e:
        print(f"Critical Error: {e}")

# Process your specific files
files_to_check = [
    ("/home/zdmaogroup/wubin/RL1-claude-optimize-memory-frame-sampling-FrPp8/checkpoints/ode_init.pt", "generator"),
    ("/home/zdmaogroup/wubin/RL1-claude-optimize-memory-frame-sampling-FrPp8/checkpoints/self_forcing_dmd.pt", "generator_ema")
]

for path, key in files_to_check:
    inspect_checkpoint(path, key)