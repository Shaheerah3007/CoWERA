import numpy as np
import mdtraj as md
from itertools import combinations


######################################## Best Hummer FNC ####################################################

#############################################################################################################
#TODO : Add provison for other features

def best_hummer_q(traj, native_file=None,  init_file=None, tar_file=None, frame_range=None):
    """
    Compute the Best Hummer Q from a trajectory and native structure.

    Parameters:
    - traj: str, md.Trajectory, or np.ndarray
        The trajectory data. Can be a file path, an md.Trajectory object, or a numpy array of coordinates.
    - native_file: str
        Path to the native structure file.
    - frame_range: tuple (start, end), optional
        Range of frames to consider from the trajectory.
    - init_file: str, optional
        Initial structure file for fallback computation.

    Returns:
    - q: np.ndarray
        Q values for each frame
    """
    #print(traj,native_file,top_file)
    try:
            # Process trajectory input
        if isinstance(traj, str):
            if native_file is None:
                traj = md.load(traj)
            else:
                traj = md.load(traj, top=native_file)
        elif isinstance(traj, md.Trajectory):
            pass  # already in proper format
        elif isinstance(traj, np.ndarray):
            traj = md.Trajectory(xyz=traj, topology=md.load(native_file).topology)
        else:
            raise TypeError("Unsupported type for traj. Must be str, md.Trajectory, or np.ndarray.")


        native = md.load(native_file)
        #print("Native structure loaded from:", native_file)
        #print("Trajectory loaded from:", traj)
        # Constants
        BETA_CONST = 50  # 1/nm
        LAMBDA_CONST = 1.8
        NATIVE_CUTOFF = 0.45  # nm

        # Compute native contacts from heavy atoms
        heavy = native.topology.select_atom_indices('heavy')
        heavy_pairs = np.array([
            (i, j) for (i, j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - native.topology.atom(j).residue.index) > 3
        ])
        heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
        native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
        #print("Number of native contacts:", len(native_contacts))

        # Slice frames if requested
        frames = traj[frame_range[0]:frame_range[1]] if frame_range is not None else traj

        # Calculate Q
        r = md.compute_distances(frames, native_contacts)
        r0 = md.compute_distances(native[0], native_contacts)
        q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
        d_range = np.array([0,1])
        return q, d_range

    except Exception as e:
    #    print(f"An error occurred during Hummer Q calculation: {e}")
        q = best_hummer_q(init_file, native_file=native_file, frame_range=frame_range)
        return q


#############################################################################################################

def RMSD_Backbone(traj, top_file=None, init_file=None, unfolded_file=None):
    """
    Compute backbone RMSD (CA atoms) using MDTraj.

    Parameters
    ----------
    traj : str, np.ndarray, or md.Trajectory
        Trajectory file path, coordinates array, or preloaded trajectory.
    top_file : str
        Topology file (PDB, PSF, etc.). Required for file or array inputs.
    native_file : str
        Reference/native structure file.

    Returns
    -------
    rmsds : np.ndarray
        RMSD values for each frame (aligned on CA atoms).
    drange : np.ndarray
        Array [0, initial RMSD].
    """
    try:
        if unfolded_file is None:
            raise ValueError("unfolded_file is required for determining rmsd range.")
        # Load or convert trajectory
        if isinstance(traj, str):
            if top_file is None:
                traj = md.load(traj)
            else:
                traj = md.load(traj, top=top_file)
        elif isinstance(traj, md.Trajectory):
            pass  # already in correct format
        elif isinstance(traj, np.ndarray):
            if top_file is None:
                raise ValueError("top_file required for NumPy coordinate array.")
            top = md.load(top_file).topology
            traj = md.Trajectory(xyz=traj, topology=top)
        else:
            raise TypeError("traj must be str, md.Trajectory, or np.ndarray.")

        # Load reference structure
        ref = md.load(top_file)


        # Get CA atom indices for alignment and RMSD
        atom_indices = traj.topology.select("name CA")

        # Align trajectory to reference on CA atoms
        traj.superpose(ref, atom_indices=atom_indices)

        # Compute RMSD on CA atoms
        rmsds = md.rmsd(traj, ref, atom_indices=atom_indices)

        # Load target structure
        tar = md.load(unfolded_file)

        # Align target to reference on CA atoms
        tar.superpose(ref, atom_indices=atom_indices)

        # Compute RMSD on CA atoms
        tar_rmsd = md.rmsd(tar, ref, atom_indices=atom_indices)[0]

        drange = np.array([0, tar_rmsd])

        return rmsds, drange

    except Exception as e:
        #print(f"Error in RMSD_Backbone: {e}")
        rmsds = RMSD_Backbone(init_file, top_file=top_file, init_file=init_file, unfolded_file=unfolded_file)
        return rmsds