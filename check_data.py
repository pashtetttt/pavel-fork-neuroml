# check_data.py
import h5py
import os
import pathlib

def check_hdf5_file(file_path):
    """
    Checks an HDF5 file for valid trials based on expected keys and attributes.
    Returns the count of valid and invalid trials.
    """
    valid_trials = 0
    invalid_trials = 0

    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                trial_valid = True
                try:
                    trial_group = f[key]

                    # Check for required datasets
                    required_datasets = ['input_features', 'seq_class_ids']
                    for ds_name in required_datasets:
                        if ds_name not in trial_group:
                            trial_valid = False
                            break

                    if not trial_valid:
                        invalid_trials += 1
                        continue

                    # Check for required attributes
                    required_attrs = ['n_time_steps', 'seq_len', 'block_num', 'trial_num']
                    for attr_name in required_attrs:
                        if attr_name not in trial_group.attrs:
                            trial_valid = False
                            break

                    if trial_valid:
                        valid_trials += 1
                    else:
                        invalid_trials += 1

                except Exception:
                    invalid_trials += 1
                    continue # This trial is invalid

    except FileNotFoundError:
        print(f"    {file_path.name}: File not found.")
    except Exception:
        print(f"    {file_path.name}: Could not open/read file.")

    return valid_trials, invalid_trials

def check_session_data(session_dir_path):
    """
    Checks both data_train.hdf5 and data_val.hdf5 in a session directory.
    Prints a summary line for the session.
    """
    session_path = pathlib.Path(session_dir_path)
    session_name = session_path.name

    # Initialize counts
    total_valid = 0
    total_invalid = 0
    train_exists = False
    val_exists = False

    # Check train file
    train_file_path = session_path / "data_train.hdf5"
    if train_file_path.exists():
        train_exists = True
        valid_train, invalid_train = check_hdf5_file(train_file_path)
        total_valid += valid_train
        total_invalid += invalid_train
    else:
        print(f"  {session_name}: Train file missing.")

    # Check val file
    val_file_path = session_path / "data_val.hdf5"
    if val_file_path.exists():
        val_exists = True
        valid_val, invalid_val = check_hdf5_file(val_file_path)
        total_valid += valid_val
        total_invalid += invalid_val
    else:
        print(f"  {session_name}: Val file missing.")

    # Print summary only if at least one file exists
    if train_exists or val_exists:
        status = "OK" if total_valid > 0 else "NO VALID TRIALS"
        print(f"  {session_name}: {status} | Train: {valid_train if train_exists else 'N/A'}, Val: {valid_val if val_exists else 'N/A'} | Valid: {total_valid}, Invalid: {total_invalid}")


if __name__ == "__main__":
    # --- CHANGE THIS PATH ---
    # Point this to your 'data/hdf5_data_final' directory
    base_data_dir = "data/hdf5_data_final" # e.g., "./data/hdf5_data_final" or "D:/path/to/data/hdf5_data_final"

    base_path = pathlib.Path(base_data_dir)

    if not base_path.exists():
        print(f"ERROR: Base data directory '{base_data_dir}' does not exist. Please check the path.")
        exit()

    # Get all subdirectories (potential session directories)
    session_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d.name.startswith('t15.20') or d.name.startswith('t12.20'))]

    if not session_dirs:
        print(f"WARNING: No session directories found in '{base_data_dir}' matching pattern 't15.20*' or 't12.20*'.")
        print("List of all items in the directory:")
        for item in base_path.iterdir():
            print(f"  - {item.name} ({'Directory' if item.is_dir() else 'File'})")
    else:
        print(f"Checking {len(session_dirs)} session directories matching pattern 't15.20*' or 't12.20*':")
        for session_dir in session_dirs:
            check_session_data(session_dir)
        print("\nCheck complete.")