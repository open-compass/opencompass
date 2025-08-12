import filecmp
import os

import fire


def compare_results(folder1, folder2, results_ignore_list=None):
    # Initialize ignore_list if not provided
    if results_ignore_list is None:
        results_ignore_list = []

    # Verify that both folders exist
    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    sub_folder1 = get_all_subpaths(folder1)[0]
    sub_folder2 = get_all_subpaths(folder2)[0]

    print('compare predicitons')
    compare_folders(os.path.join(sub_folder1, 'predictions'),
                    os.path.join(sub_folder2, 'predictions'))
    print('compare results')
    compare_folders(os.path.join(sub_folder1, 'results'),
                    os.path.join(sub_folder2, 'results'),
                    results_ignore_list=results_ignore_list)


def compare_folders(folder1, folder2, results_ignore_list=None):
    '''
    Compare the contents of files with the same name in two folders
    and their subfolders,
    ignoring files specified in the ignore_list.
    :param folder1: Path to the first folder
    :param folder2: Path to the second folder
    :param ignore_list: List of filenames to ignore
    (e.g., ['temp.txt', '.DS_Store'])
    :raises: AssertionError if any non-ignored files are missing or
    have different content
    '''
    # Initialize ignore_list if not provided
    if results_ignore_list is None:
        results_ignore_list = []

    # Verify that both folders exist
    assert os.path.isdir(folder1), f'Folder does not exist: {folder1}'
    assert os.path.isdir(folder2), f'Folder does not exist: {folder2}'

    # List to store differences found
    diff_files = []

    # Walk through the first folder and compare files
    for root, dirs, files in os.walk(folder1):
        for file in files:
            # Skip files in the ignore list
            if os.path.basename(file) in results_ignore_list:
                print('ignore case: ' + os.path.basename(file))
                continue

            # Get relative path to maintain folder structure
            rel_path = os.path.relpath(os.path.join(root, file), folder1)
            file2 = os.path.join(folder2, rel_path)

            # Check if file exists in the second folder
            if not os.path.exists(file2):
                diff_files.append((rel_path, 'File missing in second folder'))
                continue

            # Compare file content (shallow=False for content comparison)
            if not filecmp.cmp(os.path.join(root, file), file2, shallow=False):
                diff_files.append((rel_path, 'Content differs'))

    # Check for files in folder2 that don't exist in folder1
    for root, dirs, files in os.walk(folder2):
        for file in files:
            # Skip files in the ignore list
            if file in results_ignore_list:
                continue

            rel_path = os.path.relpath(os.path.join(root, file), folder2)
            file1 = os.path.join(folder1, rel_path)
            if not os.path.exists(file1):
                diff_files.append((rel_path, 'File missing in first folder'))

    # Raise AssertionError if any differences were found
    if diff_files:
        error_msg = 'Found differences in files:\n'
        error_msg += '\n'.join(f'{path}: {reason}'
                               for path, reason in diff_files)
        raise AssertionError(error_msg)


def get_all_subpaths(directory):
    '''
    Get all subpaths (files and directories) within a given directory.
    Args:
        directory (str): The root directory path to search
    Returns:
        list: A list of all complete subpaths
    '''
    subpaths = []

    # Verify the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f'Directory does not exist: {directory}')

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        # Add directories first
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            subpaths.append(full_path)

        # Then add files
        for file_name in files:
            full_path = os.path.join(root, file_name)
            subpaths.append(full_path)

    return subpaths


if __name__ == '__main__':
    fire.Fire()
