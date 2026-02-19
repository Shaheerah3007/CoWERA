from collections import defaultdict
import logging
import time
import os
import shutil
import numpy as np
from wepy.walker import WalkerState
from wepy.boundary_conditions.boundary import BoundaryConditions
from cowera.features import RMSD_Backbone, best_hummer_q

class ReceptorBC(BoundaryConditions):
    """Abstract base class for boundary conditions.

    Provides shared utilities for warping walkers to any number of
    optionally weighted initial structures through a shared
    `warp_walkers` method.

    Non-abstract implementations of this class need only implement the
    `_progress` method which should return a boolean signalling a
    warping event and the dictionary-style warping record of the
    progress for only a single walker. These records will be collated
    into a single progress record across all walkers.

    Additionally, the `_update_bc` method can be overriden to return
    'BC' group records. That method should accept the arguments shown
    in this ABC and return a list of dictionary-style 'BC' records.

    Warping of walkers with multiple initial states will be done
    according to a choice of initial states weighted on their weights,
    if given.

    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ()
    BC_SHAPES = ()
    BC_DTYPES = ()

    BC_RECORD_FIELDS = ()

    # warping fields are directly inherited

    # progress towards the boundary conditions (continual)
    PROGRESS_FIELDS = ()
    PROGRESS_SHAPES = ()
    PROGRESS_DTYPES = ()

    PROGRESS_RECORD_FIELDS = ()

    DISCONTINUITY_TARGET_IDXS = Ellipsis
    """Specifies which 'target_idxs' values are considered discontinuous targets.

    Values are either integer indices, Ellipsis (indicating all
    possible values are discontinuous), or None indicating no possible
    value is discontinuous.

    """

    def __init__(self, initial_states=None,
                 initial_weights=None,
                 target_pos=None,
                 feat=None,
                 dcd_folder=None,
                 **kwargs):
        """Base constructor for ReceptorBC.

        This should be called immediately in the subclass `__init__`
        method.

        If the initial weights for each initial state are not given
        uniform weights are assigned to them.

        Arguments
        ---------
        initial_states : list of objects implementing the State interface
            The list of possible states that warped walkers will assume.

        initial_weights : list of float, optional
            List of normalized probabilities of the initial_states
            provided. If not given, uniform probabilities will be
            used.

        idxs : arraylike of int
            The indices of the atom positions in the state.

        Raises
        ------
        AssertionError
            If any of the following kwargs are not given:
            initial_states, idxs, target_pos, eigenvecs, feat.

        """



        self._initial_states = initial_states
        self._pos = target_pos
        self._feat = feat
        self.dcd_folder = dcd_folder
        # we want to choose initial states conditional on their
        # initial probability if specified. If not specified assume
        # assume uniform probabilities.
        if initial_weights is None:
            self._initial_weights = [1/len(initial_states) for _ in initial_states]
        else:
            self._initial_weights = initial_weights

    @property
    def initial_states(self):
        """The possible initial states warped walkers may assume."""
        return self._initial_states

    @property
    def initial_weights(self):
        """The probabilities of each initial state being chosen during a warping."""
        return self._initial_weights

    @property
    def idxs(self):
        """The indices of the atom positions in the state."""
        return self._idxs

    @property
    def target_pos(self):
        """Target state coordinate."""

        return self._pos

    @property
    def eigenvecs(self):
        """Eigenvectors."""
        return self._eigenvectors

    @property
    def feat(self):
        """Feature selection."""
        return self._feat


    def _progress(self, walker):
        """The method that must be implemented in non-abstract subclasses.

        Should decide if a walker should be warped or not and what its
        progress is regardless.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        to_warp : bool
           Whether the walker should be warped or not.

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        raise NotImplementedError

    def _warp(self, walker):
        """Perform the warping of a walker.

        Chooses an initial state to replace the walker's state with
        according to it's given weight.

        Returns a walker of the same type and weight.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        warped_walker : object implementing the Walker interface
            The walker with the state after the warping. Weight should
            be the same.

        warping_data : dict of str : value
           The dictionary-style 'WARPING' record for this
           event. Excluding the walker index which is done in the main
           `warp_walkers` method.

        """


        # choose a state randomly from the set of initial states
        target_idx = np.random.choice(range(len(self.initial_states)), 1,
                                  p=self.initial_weights/np.sum(self.initial_weights))[0]

        warped_state = self.initial_states[target_idx]

        # set the initial state into a new walker object with the same weight
        warped_walker = type(walker)(state=warped_state, weight=walker.weight)

        # the data for the warp
        warp_data = {'target_idx' : np.array([target_idx]),
                     'weight' : np.array([walker.weight])}

        return warped_walker, warp_data


    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """Perform an update to the boundary conditions.

        No updates to the bc are ever done in this null
        implementation.

        Parameters
        ----------
        new_walkers : list of walkers
            The walkers after warping.

        warp_data : list of dict

        progress_data : dict

        cycle : int

        Returns
        -------
        bc_data : list of dict
            The dictionary-style records for BC update events

        """

        # do nothing by default
        return []


    def warp_walkers(self, walkers, cycle):
        """Test the progress of all the walkers, warp if required, and update
        the boundary conditions.

        Arguments
        ---------

        walkers : list of objects implementing the Walker interface

        cycle : int
            The index of the cycle.

        Returns
        -------

        new_walkers : list of objects implementing the Walker interface
            The new set of walkers that may have been warped.

        warp_data : list of dict of str : value
            The dictionary-style records for WARPING update events


        bc_data : list of dict of str : value
            The dictionary-style records for BC update events

        progress_data : dict of str : arraylike
            The dictionary-style records for PROGRESS update events

        """

        new_walkers = []
        dcd_folder = self.dcd_folder
        # sporadic, zero or many records per call
        warp_data = []
        bc_data = []

        # continual, one record per call
        progress_data = defaultdict(list)

        # calculate progress data
        all_progress_data = [self._progress(w) for w in walkers]

        for walker_idx, walker in enumerate(walkers):

            # unpack progress data
            to_warp, walker_progress_data = all_progress_data[walker_idx]

            # add that to the progress data record
            for key, value in walker_progress_data.items():
                progress_data[key].append(value)

            # if the walker is meets the requirements for warping warp
            # it
            if to_warp:
                # Warp the walker
                warped_walker, walker_warp_data = self._warp(walker)
                walker_warp_data['walker_idx'] = np.array([walker_idx])
                new_walkers.append(warped_walker)
                warp_data.append(walker_warp_data)

                # Save productive trajectory
                try:
                    walker_dcd_file = os.path.join(dcd_folder, f'walker_{walker_idx}.dcd')
                    existing = [f for f in os.listdir(dcd_folder) if f.startswith("productive_traj_") and f.endswith(".dcd")]
                    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing if f.split('_')[-1].split('.')[0].isdigit()]
                    next_index = max(indices) + 1 if indices else 0
                    productive_file = os.path.join(dcd_folder, f'productive_traj_{next_index}.dcd')
                    #print(f"Copying {walker_dcd_file} to {productive_file}")
                    #print("Checking walker DCD:", walker_dcd_file, "Exists?", os.path.isfile(walker_dcd_file))
                    #print("Checking productive file:", productive_file, "Exists?", os.path.isfile(productive_file))
                    if os.path.isfile(walker_dcd_file):
                        # Copy the walker DCD file to the productive trajectory file
                        shutil.copy(walker_dcd_file, productive_file)
                        # Truncate the original file to clear it
                        os.remove(walker_dcd_file)
                        logging.info(f"Saved productive trajectory: {productive_file}")
                    else:
                        logging.warning(f"Walker DCD file not found: {walker_dcd_file}")
                except Exception as e:
                    logging.error(f"Error copying productive trajectory for walker {walker_idx}: {e}")

                logging.info(f'WARP EVENT observed at {cycle}')
                logging.info(f'Warped Walker Weight = {walker_warp_data["weight"]}')

            # no warping so just return the original walker
            else:
                new_walkers.append(walker)

        # consolidate the progress data to an array of a single
        # feature vectors for the cycle
        for key, value in progress_data.items():
            progress_data[key] = value

        # if the boundary conditions need to be updated given the
        # cycle and state from warping perform that now and return any
        # record data for that
        bc_data = self._update_bc(new_walkers, warp_data, progress_data, cycle)
        #print(f"Cycle {cycle} new walkers: {new_walkers}")

        return new_walkers, warp_data, bc_data, progress_data

    @classmethod
    def warping_discontinuity(cls, warping_record):
        """Tests whether a warping record generated by this class is
        discontinuous or not.

        Parameters
        ----------

        warping_record : tuple
            The WARPING type record.

        Returns
        -------

        is_discontinuous : bool
            True if a discontinuous warp False if continuous.

        """

        # if it is Ellipsis then all possible values are discontinuous
        if cls.DISCONTINUITY_TARGET_IDXS is Ellipsis:
            return True

        # if it is None then all possible values are continuous
        elif cls.DISCONTINUITY_TARGET_IDXS is None:
            return False

        # otherwise it will have a tuple of indices for the
        # target_idxs that are discontinuous targets
        elif warping_record[2] in cls.DISCONTINUITY_TARGET_IDXS:
            return True

        # otherwise it wasn't a discontinuous target
        else:
            return False

#####################################################################################################
#######################################################################################################
#######################################################################################################

class TargetBC(ReceptorBC):
    """Boundary condition for wrapping walkers.

    Walkers will be warped (discontinuously) if a walker image is at least
    a certain distance away from the target state image.

    Wraping will replace the walker state with the initial state given
    as a parameter to this class.


    """

    # records of boundary condition changes (sporadic)
    BC_FIELDS = ReceptorBC.BC_FIELDS + ('boundary_distance', )
    """
    Only occurs at the start of the simulation and just reports on the
    cutoff distance.
    """

    BC_SHAPES = ReceptorBC.BC_SHAPES + ((1,), )
    BC_DTYPES = ReceptorBC.BC_DTYPES + (float, )
    BC_RECORD_FIELDS = ReceptorBC.BC_RECORD_FIELDS + ('boundary_distance', )

    # warping (sporadic)
    WARPING_FIELDS = ReceptorBC.WARPING_FIELDS + ()
    WARPING_SHAPES = ReceptorBC.WARPING_SHAPES + ()
    WARPING_DTYPES = ReceptorBC.WARPING_DTYPES + ()

    WARPING_RECORD_FIELDS = ReceptorBC.WARPING_RECORD_FIELDS + ()

    # progress record group
    PROGRESS_FIELDS = ReceptorBC.PROGRESS_FIELDS + ('distances_fromtarget',)
    """
    The 'distances_fromtarget' field reports on the distance for each walker from target.

    """

    PROGRESS_SHAPES = ReceptorBC.PROGRESS_SHAPES + (Ellipsis,)
    PROGRESS_DTYPES = ReceptorBC.PROGRESS_DTYPES + (float,)
    PROGRESS_RECORD_FIELDS = ReceptorBC.PROGRESS_RECORD_FIELDS + ('distances_fromtarget', )

    def __init__(self, initial_state=None,
                 cutoff_distance=None,
                 target_pos=None,
                 feat=None,
                 dcd_folder=None,
                 native_file=None,
                 init_file=None,
                 tar_file=None,
                 top_file=None,
                 **kwargs):
        """Constructor for TargetBC class.

        All the key-word arguments are necessary.

        The 'initial_state' should be the initial state of your
        simulation for proper non-equilibrium simulations.

        Arguments
        ---------
        initial_state : object implementing State interface
            The state walkers will take on after reaching to
            the target state.

        cutoff_distance : float
            The distance that specifies the boundary condition. When
            the distance from target is less than this, the walker will
            be wraped.

        idxs : list of int
           Indices of the atoms in the topology that.

        Raises
        ------
        AssertionError
            If any of the following are not provided: initial_state, idxs,
            target_pos, eigenvecs and feat.

        AssertionError
            If the cutoff distance is not a float.

        Warnings
        --------
        The 'initial_state' should be the initial state of your
        simulation for proper non-equilibrium simulations.

        """

        # since the super class can handle multiple initial states we
        # wrap the single initial state to a list.
        super().__init__(initial_states=[initial_state],
                         target_pos=target_pos,
                         **kwargs)

        # test input
        assert type(cutoff_distance) is float

        self._cutoff_distance = cutoff_distance
        self.dcd_folder = dcd_folder
        self.sel_feat = feat
        self.native_path = native_file
        self.init_file = init_file
        self.tar_file = tar_file
        self.top_file = top_file
    @property
    def cutoff_distance(self):
        """The distance from target a walker must be to be wrapped."""
        return self._cutoff_distance

    def _calc_distance_fromtarget(self, walker):
        """Distance for a walker from target state.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        distance_fromtarget : float

        """

        if self.sel_feat == "best_hummer_q":
            # Calculate the best_hummer_q projection coordinates
            proj_coord,_ = best_hummer_q(
                traj=walker.state['positions'],
                native_file=self.native_path,
            )
        elif self.sel_feat == "rmsd_backbone":
            proj_coord, _ = RMSD_Backbone(
                traj=walker.state['positions'],
                top_file=self.top_file,
                init_file=self.init_file,
                unfolded_file=self.tar_file # TODO: doesnt matter give a dummy file we are not interested in the range so it may or may not be actual unfolded
            )
        else:
            raise ValueError("Unrecognized feature selection")

        distance_fromtarget = np.linalg.norm(proj_coord - self._pos)
        #print(f"Distance from target: {distance_fromtarget}")
        return distance_fromtarget

    def _progress(self, walker):
        """Calculate whether a walker has reached the target and also provide a
        dictionary for a single walker in the progress records.

        Parameters
        ----------
        walker : object implementing the Walker interface

        Returns
        -------
        is_reached : bool
           Whether the walker is reached (warped) or not

        progress_data : dict of str : value
           Dictionary of the progress record group fields
           for this walker alone.

        """

        distance_fromtarget = self._calc_distance_fromtarget(walker)
        #print("Distance from target:", distance_fromtarget)
        # test to see if the walker is reached
        reached = False
        if distance_fromtarget <= self._cutoff_distance:
            reached = True
        #print(f"cutoff distance: {self._cutoff_distance}")
        progress_data = {'distances_fromtarget' : distance_fromtarget}

        return reached, progress_data

    def _update_bc(self, new_walkers, warp_data, progress_data, cycle):
        """Perform an update to the boundary conditions.

        This is only used on the first cycle to keep a record of the
        cutoff parameter.

        Parameters
        ----------
        new_walkers : list of walkers
            The walkers after warping.

        warp_data : list of dict

        progress_data : dict

        cycle : int

        Returns
        -------
        bc_data : list of dict
            The dictionary-style records for BC update events

        """

        # Only report a record on
        # the first cycle which gives the distance at which walkers
        # are warped
        if cycle == 0:
            return [{'boundary_distance' : np.array([self._cutoff_distance]),},]
        else:
            return []
