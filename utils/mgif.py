import os
import re
import imageio.v2 as imageio

def natural_sort_key(filename):
    parts = re.split(r'(\d+)', filename)
    return [int(part) if part.isdigit() else part for part in parts]

def make_gif(folder_name, name, duration=0.03):
    os.makedirs('./public/gif', exist_ok=True)
    output_dir = os.path.join('./public/gif', f'{name}.gif')
    train_folder = os.path.join(folder_name, 'train')
    files = sorted(
        (f for f in os.listdir(train_folder)
         if f.lower().endswith(('.png')) and not f.lower().endswith('_normal.png')),
        key=natural_sort_key
    )
    images = [
        imageio.imread(os.path.join(train_folder, f))[:, :, :3]
        for f in files
    ]
    imageio.mimsave(output_dir, images, duration=duration, loop=0)
    print(f'saved {name}.gif to {output_dir} ({len(images)} frames @ {duration}s/frame)')

def main():
    for folder in os.listdir('./output/'):
        item_path = os.path.join('./output', folder)
        if os.path.isdir(item_path):
            make_gif(item_path, folder, duration=0.03)

if __name__ == '__main__':
    main()