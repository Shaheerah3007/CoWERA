import os
import shutil

def update_dcd_files(resampling_data, dcd_folder=".", prefix="walker_", ext=".dcd"):
    """
    Rename, copy, and delete walker DCD files based on resampling decisions.

    Parameters
    ----------
    resampling_data : list of dict
        Each dict must include 'walker_idx' (int), 'target_idxs' (array), and 'decision_id' (array).
    dcd_folder : str
        Directory where DCD files are stored.
    prefix : str
        Prefix for walker DCD files.
    ext : str
        File extension for DCD files.
    """
    #print(resampling_data)
    new_files = set()
    active_ids = set()
    clone_pairs = []  # (source_id, dest_id)

    for record in resampling_data:
        source_id = int(record["walker_idx"][0])
        target_ids = record["target_idxs"]
        decision_id = int(record["decision_id"][0])
        #print(f"source_id: {source_id}, target_ids: {target_ids}, decision_id: {decision_id}")
        for target_id in target_ids:
            target_id = int(target_id)
            # Handle different decisions

            if decision_id == 1:#ResamplingDecision.NOTHING
                # Keep walker; if target differs, it's a clone operation
                if source_id != target_id:
                    clone_pairs.append((source_id, target_id))
                active_ids.add(target_id)


            elif decision_id == 2:#ResamplingDecision.CLONE
                if source_id != target_id:
                    clone_pairs.append((source_id, target_id))
                active_ids.add(target_id)

            elif decision_id == 3: #ResamplingDecision.KEEP_MERGE
                # Keep the merged-into walker
                active_ids.add(source_id)

            elif decision_id == 4:#ResamplingDecision.SQUASH
                # Walker is discarded (squashed into another)
                continue

    #print(f"clone_pairs: {clone_pairs}")
    # Perform all cloning
    for src_id, dest_id in clone_pairs:
        src_file = os.path.join(dcd_folder, f"{prefix}{src_id}{ext}")
        dest_file = os.path.join(dcd_folder, f"{prefix}{dest_id}{ext}")
        if os.path.exists(src_file):
            try:
                shutil.copy(src_file, dest_file)
                new_files.add(f"{prefix}{dest_id}{ext}")
            except Exception as e:
                print(f"Error copying {src_file} to {dest_file}: {e}")
        else:
            print(f"Warning: source file {src_file} does not exist.")

    # Add all active walker files (even if not cloned)
    for idx in active_ids:
        new_files.add(f"{prefix}{idx}{ext}")

    # Find current walker files
    existing_files = {fname for fname in os.listdir(dcd_folder)
                      if fname.startswith(prefix) and fname.endswith(ext)}

    # Delete unneeded walker files
    #to_delete = existing_files - new_files
    #for fname in to_delete:
    #    path = os.path.join(dcd_folder, fname)
    #    os.remove(path)
        #print(f"Deleted: {path}")
