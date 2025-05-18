import os
import zipfile

base_path = '/scratch-shared/scur0551/GenImage_download'
zip_file = os.path.join(base_path, 'GenImage_restored.zip')

# datasets = ['ADM', 'BigGAN', 'Midjourney', 'stable_diffusion_v_1_5']
datasets = ['VQDM']

valid_paths = []
for dataset in datasets:
    if dataset == 'stable_diffusion_v_1_5':
        valid_paths.extend([(dataset, 'train'), (dataset, 'val')])
    else:
        valid_paths.append((dataset, 'val'))

target_subdirs = {
    f'GenImage/{dataset}/{target}/ai/': f'{base_path}/GenImage/{dataset}/{target}/ai_og'
    for dataset, target in valid_paths
}

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if not any(member.startswith(f'GenImage/{dataset}/') for dataset in datasets):
            continue
        for subdir, out_path in target_subdirs.items():
            if member.startswith(subdir) and not member.endswith('/'):
                relative_path = member[len(subdir):]
                dest_path = os.path.join(out_path, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with zip_ref.open(member) as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                break