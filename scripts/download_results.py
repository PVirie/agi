import argparse
import os
import subprocess
import dotenv

dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(os.path.join(dir_path, '../secrets.env'))

GPU_SERVER_PATH = os.getenv('GPU_SERVER_PATH')

artifacts_path = os.path.join(dir_path, '../artifacts')

SYNC_FOLDERS = [
    'experiments'
]


def sync_folder(folder: str, override: bool):
    """Download a subfolder from the remote GPU server to the local artifacts directory.
    Remote is always the source; local is always the destination — no upload can occur.
    """
    remote = f'{GPU_SERVER_PATH}artifacts/{folder}/'
    local = os.path.join(artifacts_path, folder)
    os.makedirs(local, exist_ok=True)
    extra_flags = ['--checksum'] if override else ['--ignore-existing']
    result = subprocess.run(
        [
            'rsync',
            '-avz',
            *extra_flags,
            '--no-perms',
            '-e', 'ssh -o StrictHostKeyChecking=no',
            remote,       # source: remote server
            local + '/',  # destination: local directory
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f'Warning: rsync for {folder!r} exited with code {result.returncode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sync artifacts from the GPU server.')
    parser.add_argument(
        '-f', '--override',
        action='store_true',
        default=False,
        help='Overwrite local files that differ from the server (default: skip existing files).',
    )
    args = parser.parse_args()

    os.makedirs(artifacts_path, exist_ok=True)

    for folder in SYNC_FOLDERS:
        print(f'Syncing {folder} (override={args.override})...')
        sync_folder(folder, override=args.override)

    print('Done.')
