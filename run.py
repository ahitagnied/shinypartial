import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from scripts.single_obj import load_config, render_obj

def main():
    config_path = os.path.join(current_dir, "configs", "single_obj.yaml")
    print(f"looking for config at: {config_path}")
    if os.path.exists(config_path):
        print("config file found!")
    else:
        print("config file NOT found!")
        config = load_config(config_path)

    render_obj(config, train_ratio=0.33)

if __name__ == "__main__":
    main()