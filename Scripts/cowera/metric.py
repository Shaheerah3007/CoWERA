import numpy as np
import ruptures as rpt
from cowera.features import best_hummer_q, RMSD_Backbone
from joblib import Parallel, delayed
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSD, rmsd

class Calculate_Distances:
    def __init__(self, feat, increment, native_file=None, init_file=None, tar_file=None, top_file=None, distance_criterion="pairwise_rmsd"):
        super().__init__()
        self._feat = feat
        self.native_file = native_file
        self.init_file = init_file
        self.tar_file = tar_file
        self.top_file = top_file
        self.increment = increment
        self.distance_criterion = distance_criterion

    def image_distance(self, state1, state2):
        if self.distance_criterion == "pairwise_rmsd":
            try:
                coord1 = state1['positions']
                coord2 = state2['positions']

                # Create two universes from positions + topology
                u1 = mda.Universe(self.native_file)
                u2 = mda.Universe(self.native_file)

                u1.load_new(coord1, order='fac')
                u2.load_new(coord2, order='fac')

                rmsd_val = rmsd(u1.select_atoms("backbone").positions, u2.select_atoms("backbone").positions, center=True, superposition=True)
                #print("RMSD is", rmsd_val)
                return rmsd_val

            except Exception as e:
                print(f"RMSD computation failed: {e}")
                return None
        elif self.distance_criterion == "euclidean":
            try:

                image1 = self.get_proj_coord(state1)
                image2 = self.get_proj_coord(state2)

                return np.linalg.norm(image1 - image2)

            except Exception as e:
                print(f"Euclidean distance computation failed: {e}")
                return None
        else:
            raise ValueError(f"Unrecognized distance criterion: {self.distance_criterion} use \"pairwise_rmsd\" or \"euclidean\"")

    def get_proj_coord(self, state):
        """
        Compute the 'image' of a walker state — e.g., projecting to a collective variable.
        """

        if self._feat == "best_hummer_q":
            # Compute the best Hummer Q value from positions
            q,_ = best_hummer_q(
                traj=state['positions'],
                native_file=self.native_file,
            )
            proj_coord = q

        elif self._feat == "rmsd_backbone":
            # Compute the RMSD backbone from positions
            if self.increment == 1:
                rmsd, _ = RMSD_Backbone(
                    traj=state['positions'],
                    top_file=self.top_file,
                    init_file=self.init_file,
                    unfolded_file=self.tar_file
                )
            else:
                rmsd, _ = RMSD_Backbone(
                    traj=state['positions'],
                    top_file=self.top_file,
                    init_file=self.init_file,
                    unfolded_file=self.init_file
                )
            proj_coord = rmsd
        else:
            raise ValueError(f"Unrecognized feature selection: {self._feat}")
        #print(f"Projected coordinate for {self._feat}: {proj_coord}")
        return proj_coord

    def get_bin_edges(self,x_min = 0, x_max=1, n_bins=100):
        return np.linspace(x_min, x_max, n_bins + 1)

    def get_bin(self, x, bin_edges):
        return np.digitize(x, bin_edges)


    def phase_calculation(self, i, path, n_d,n_bins=100):
        if self._feat == "best_hummer_q":
            projection, drange = best_hummer_q(
                traj=f"{path}walker_{i}.dcd",
                native_file=self.native_file,
                init_file=self.init_file,
            )
        elif self._feat == "rmsd_backbone":
            if self.increment == 1:
                projection, drange = RMSD_Backbone(
                    traj=f"{path}walker_{i}.dcd",
                    init_file=self.init_file,
                    unfolded_file=self.tar_file,
                    top_file=self.top_file
                )
            else:
                projection, drange = RMSD_Backbone(
                    traj=f"{path}walker_{i}.dcd",
                    init_file=self.init_file,
                    unfolded_file=self.init_file,
                    top_file=self.top_file
                )

        else:
            raise ValueError(f"Unrecognized feature selection: {self._feat}")

        if len(projection) == 1:
            #print(f"Walker {i}: has been warped.")
            return np.zeros(n_d), 0.0 , projection[0]

        drange = drange[::self.increment]

        # Fast changepoints
        algo = rpt.KernelCPD(kernel="linear",min_size=2).fit(projection)
        changes = np.array(algo.predict(pen=0.01))
        # print("Changes are", changes, "n_d is", n_d)
        change_points = np.concatenate(([0], changes[:-1] - n_d, changes[-1:])) if  np.all(changes[:-1] - n_d) > 0 else np.concatenate(([0], changes))
        # print("Change points are", change_points)
        # Weight (normalized projection assumed)
        weight = projection[-1]
        # projection[change_points[-2]]

        bin_edges = self.get_bin_edges(x_min=drange[0], x_max=drange[1], n_bins=n_bins)

        # Digitize instead of loop
        current_bins = np.digitize(projection, bin_edges, right=True)

        # Fast phase calculation
        seg = current_bins[change_points[-2]:change_points[-1]]
        phase = np.sign(np.diff(seg)).mean()
        #print(f"Walker {i}: Phase {phase}, Weight {weight}, Bins {current_bins}")
        # print(f"Walker {i}: Phase {phase}, Weight {weight}, Bins {current_bins}, projection: {projection}")
        return current_bins[-n_d:], phase, weight


    def intensity_calculation(self, n_walkers, path,n_d,it,
                        n_bins=100, max_bins=125,
                        bin_increase_factor=1.2, bin_decrease_factor=0.8):
        """
        Compute new weights for walkers based on collective motion and phase,
        and adjust bin resolution based on collective bin movement.
        """
        #load bin edges
        # Cached bin edges
        if not hasattr(self, "_cached_bins") or self._cached_bins["n_bins"] != n_bins:
            self._cached_bins = {
                "n_bins": n_bins,
            }
        n_bins = self._cached_bins["n_bins"]
        #print(f"Cycle {it}: Using {n_bins} bins.")


        bins_list, phase_arr, weight_arr = [], [], []
        for i in range(n_walkers):
            bins, phase, weight = self.phase_calculation(i, path,n_d, n_bins)
            bins_list.append(bins)
            phase_arr.append(phase)
            weight_arr.append(weight)


        bins_arr = np.array(bins_list, dtype=float)

        #mask = ~np.isnan(bins_arr).any(axis=1)
        #print("mask is", mask)
        if (bins_arr == 0).all():
            print(f"Cycle {it}: no valid walkers, uniform weights.")
            return np.ones(n_walkers) / n_walkers, n_bins

        #valid_bins = bins_arr[mask].astype(int)
        phases = np.array(phase_arr).astype(float)
        weights = np.array(weight_arr).astype(float)
        weights_scaled = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights_scaled = np.where(self.increment == 1, weights_scaled, 1 - weights_scaled)


        pmin = np.min(phases)
        pmax = np.max(phases)
        phases_scaled = 2.0 * (phases - pmin) / (pmax - pmin) - 1.0

        fraction_unique = np.mean([
            len(np.unique(bins_arr[i])) / bins_arr.shape[1]
            for i in range(bins_arr.shape[0])
        ])
        if fraction_unique < 0.2:
            #print(f"Cycle {it}: Increasing bin resolution")
            n_bins = min(max_bins, int((n_bins - 1) * bin_increase_factor))
        elif fraction_unique > 0.8:
            #print(f"Cycle {it}: Decreasing bin resolution")
            n_bins = max(10, int((n_bins - 1) * bin_decrease_factor))

        interference = weights_scaled * np.sqrt(1 + phases_scaled)

        Intensity = interference / interference.max()

        return Intensity, n_bins








