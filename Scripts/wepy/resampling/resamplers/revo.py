
import random as rand
import itertools as iter

import logging
from eliot import start_action, log_call

import numpy as np

from wepy.resampling.resamplers.resampler import Resampler
from wepy.resampling.resamplers.clone_merge  import CloneMergeResampler
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision
from File_resampler import update_dcd_files

class REVOResampler(CloneMergeResampler):
    r"""

    Resampler implementing the new REVO algorithm.

    """

    # fields for resampler data
    RESAMPLING_FIELDS = CloneMergeResampler.RESAMPLING_FIELDS
    RESAMPLING_SHAPES = CloneMergeResampler.RESAMPLING_SHAPES #+ (Ellipsis,)
    RESAMPLING_DTYPES = CloneMergeResampler.RESAMPLING_DTYPES #+ (np.int,)


    # fields that can be used for a table like representation
    RESAMPLING_RECORD_FIELDS = CloneMergeResampler.RESAMPLING_RECORD_FIELDS

    # fields for resampling data
    RESAMPLER_FIELDS = CloneMergeResampler.RESAMPLER_FIELDS + \
                       ('num_walkers', 'distance_array', 'variation', 'image_shape', 'images')
    RESAMPLER_SHAPES = CloneMergeResampler.RESAMPLER_SHAPES + \
                       ((1,), Ellipsis, (1,), Ellipsis, Ellipsis)
    RESAMPLER_DTYPES = CloneMergeResampler.RESAMPLER_DTYPES + \
                       (int, float, float, int, None)

    # fields that can be used for a table like representation
    RESAMPLER_RECORD_FIELDS = CloneMergeResampler.RESAMPLER_RECORD_FIELDS + \
                              ('variation',)


    def __init__(self,
                 distance=None,
                 run_id=None,
                 merge_dist=None,
                 pmax=0.20,
                 pmin=1e-12,
                 init_state=None,
                 seed=None,
                 path=None,
                 increment = None,
                 dcd_folder = None,
                 n_d = None,
                 **kwargs):

        """Constructor for the REVO Resampler.

        Parameters
        ----------


        distance : object implementing Distance
            The distance metric to compare walkers.

        merge_dist : float
            The merge distance threshold. Units should be the same as
            the distance metric.

        init_state : WalkerState object
            Used for automatically determining the state image shape.

        seed : None or int, optional
            The random seed. If None, the system (random) one will be used.

        """

        # call the init methods in the CloneMergeResampler
        # superclass. We set the min and max number of walkers to be
        # constant
        super().__init__(pmin=pmin, pmax=pmax,
                         min_num_walkers=Ellipsis,
                         max_num_walkers=Ellipsis,
                         **kwargs)

        assert distance is not None,  "Distance object must be given."
        assert init_state is not None,  "An  state must be given."

        # Directory

        self.path = path

        # the distance metric

        self.distance = distance

        # merge distance

        self.merge_dist = merge_dist

        # run index

        self.run_id = run_id

        # direction

        self.increment = increment

        # number of data per sement

        self.n_d = n_d

        # setting the random seed
        self.seed = seed

        #dcd folder
        self.dcd_folder = dcd_folder
        if seed is not None:
            rand.seed(seed)

        # we do not know the shape and dtype of the images until
        # runtime so we determine them here
        #print("init state",init_state)
        image = self.distance.image(init_state)
        self.image_dtype = image.dtype

    def resampler_field_dtypes(self):
        """ Finds out the datatype of the image.

        Returns
        -------
        datatypes : tuple of datatype
        The type of reasampler image.

        """

        # index of the image idx
        image_idx = self.resampler_field_names().index('images')

        # dtypes adding the image dtype
        dtypes = list(super().resampler_field_dtypes())
        dtypes[image_idx] = self.image_dtype

        return tuple(dtypes)


    def _calcvariation(self, num_walker_copies, distance_arr,walker_weights):

        #print("num_walker_copies",num_walker_copies)
        #print("distance_arr",distance_arr)

        num_walkers = len(distance_arr)


        # the walker variation values (Vi values) and variation
        walker_variations = np.zeros(num_walkers)
        variation = 0.0


        # calculate  walker variation values

        for i in range(num_walkers):
            if num_walker_copies[i] > 0:
                walker_variations[i] = distance_arr[i]* num_walker_copies[i] * walker_weights[i]
                variation += distance_arr[i] * num_walker_copies[i] * walker_weights[i] # instead of distance arr there will be my weight array
        #print("walker_variations",walker_variations)
        #print("variation",variation)
        return variation, walker_variations

    def decide(self, walker_weights, num_walker_copies, distance_arr, distance_matrix):
        """
        Optimize the trajectory variation by making decisions for resampling.

        """

        num_walkers = len(walker_weights)

        variations = []
        merge_groups = [[] for i in range(num_walkers)]
        walker_clone_nums = [0 for i in range(num_walkers)]

        # make copy of walkers properties
        new_walker_weights = walker_weights.copy()
        new_num_walker_copies = num_walker_copies.copy()


        # calculate the initial variation which will be optimized
        variation, walker_variations = self._calcvariation(new_num_walker_copies,
                                                           distance_arr,walker_weights)
        variations.append(variation)

        # maximize the variance through cloning and merging
        logging.info("Starting variance optimization: {}".format(variation))

        productive = True
        while productive:
            productive = False
            # find min and max walker_variationss, alter new_amp

            # initialize to None, we may not find one of each
            min_idx = None
            max_idx = None

            # selects a walker with minimum walker_variations and a walker with
            # maximum walker_variations walker (distance to other walkers) will be
            # tagged for cloning (stored in maxwind), except if it is
            # already a keep merge target
            max_tups = []
            for i, value in enumerate(walker_variations):
                # 1. must have an amp >=1 which gives the number of clones to be made of it
                # 2. must not already be a keep merge target
                if (new_num_walker_copies[i] >= 1) and \
                   (new_walker_weights[i]/(new_num_walker_copies[i] + 1) > self.pmin) and \
                   (len(merge_groups[i]) == 0):
                    max_tups.append((value, i))


            if len(max_tups) > 0:
                max_value, max_idx = max(max_tups)

            # walker with the lowest walker_variations
            # will be tagged for merging (stored in min_idx)
            min_tups = [(value, i) for i,value in enumerate(walker_variations)
                        if (new_num_walker_copies[i] == 1) and (new_walker_weights[i] < self.pmax)  and (np.sum(distance_matrix[i] <= self.merge_dist) >= 1)]

            if len(min_tups) > 0:
                min_value, min_idx = min(min_tups)

            # does min_idx have an eligible merging partner?
            closewalk = None
            condition_list = np.array([i is not None for i in [min_idx, max_idx]])
            if condition_list.all() and min_idx != max_idx:

                # get the walkers that aren't the minimum and the max
                # walker_variations walkers, as candidates for merging
                closewalks = set(range(num_walkers)).difference([min_idx, max_idx])

                # remove those walkers that if they were merged with
                # the min walker_variations walker would violate the pmax
                # and walkers should be away from target
                closewalks = [idx for idx in closewalks
                                      if (new_num_walker_copies[idx]==1) and
                                       (new_walker_weights[idx] + new_walker_weights[min_idx] < self.pmax)]

                # if there are any walkers left, create a list of them
                if len(closewalks) > 0:
                    closewalks_dists = [(distance_matrix[min_idx][i], i) for i in closewalks if distance_matrix[min_idx][i] <= (self.merge_dist)]

                    # if any were found set this as the closewalk
                    if len(closewalks_dists) > 0:
                        closedist, closewalk = random.choice(closewalks_dists)


            # did we find a closewalk?
            condition_list = np.array([i is not None for i in [min_idx, max_idx, closewalk]])
            #if we find a walker for cloning, a walker and its close neighbor for merging
            if condition_list.all() :

                # change new_amp
                #r = rand.uniform(0.0, new_walker_weights[closewalk] + new_walker_weights[min_idx])
                #if (r < new_walker_weights[closewalk]):
                keep_idx = closewalk
                squash_idx = min_idx
                #else:
                #    keep_idx = min_idx
                #    squash_idx = closewalk

                # update new_num_walker_copies
                new_num_walker_copies[squash_idx] = 0
                new_num_walker_copies[keep_idx] = 1
                new_num_walker_copies[max_idx] += 1

                # re-determine variation function, and walker_variations values
                new_variation, walker_variations = self._calcvariation(new_num_walker_copies, distance_arr,walker_weights)

                if new_variation > variation: # resampling is possible with these choices of walkers
                    variations.append(new_variation)

                    logging.info("Variance move to {} accepted".format(new_variation))

                    productive = True
                    variation = new_variation

                    # update weight
                    new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                    new_walker_weights[squash_idx] = 0.0

                    # add the squash index to the merge group
                    merge_groups[keep_idx].append(squash_idx)

                    # add the indices of the walkers that were already
                    # in the merge group that was just squashed
                    merge_groups[keep_idx].extend(merge_groups[squash_idx])

                    # reset the merge group that was just squashed to empty
                    merge_groups[squash_idx] = []

                    # increase the number of clones that the cloned
                    # walker has
                    walker_clone_nums[max_idx] += 1

                    logging.info("variance after selection: {}".format(new_variation))

                # if not productive
                else:
                    new_num_walker_copies[min_idx] = 1
                    new_num_walker_copies[closewalk] = 1
                    new_num_walker_copies[max_idx] -= 1

        # given we know what we want to clone to specific slots
        # (squashing other walkers) we need to determine where these
        # squashed walkers will be merged
        walker_actions = self.assign_clones(merge_groups, walker_clone_nums)

        # because there is only one step in resampling here we just
        # add another field for the step as 0 and add the walker index
        # to its record as well
        for walker_idx, walker_record in enumerate(walker_actions):
            walker_record['step_idx'] = np.array([0])
            walker_record['walker_idx'] = np.array([walker_idx])

        if (variations[-1] > variations[0]):
            happen = True
        else:
            happen = False

        return walker_actions, variations[-1], happen

    def get_dist(self, walkers, folder, increment, n_d, it, n_bins, max_bins):

        # initialize arrays
        dl = np.zeros(len(walkers))
        dist_mat = np.zeros((len(walkers), len(walkers)))

        # make images for all the walker states
        images = []
        for walker in walkers:
            image = self.distance.image(walker.state)
            images.append(image)

        # get the combinations of indices for all walker pairs
        for i, j in iter.combinations(range(len(images)), 2):

            # calculate
            d12 = self.distance.image_distance(images[i], images[j])
            dist_mat[i][j] = d12
            dist_mat[j][i] = d12

        dl, n_bins = self.distance.intensity_calculation(n_walkers= len(walkers) ,path = folder, increment = increment,n_d = n_d, it = it, n_bins = n_bins,max_bins = max_bins)

        return dl, [walker_dists for walker_dists in dist_mat], images , n_bins

    @log_call(include_args=[],
              include_result=False)

    def resample(self, walkers, cycle_id, n_bins, max_bins=125):
        """Resamples walkers based on REVO algorithm

        Parameters
        ----------
        walkers : list of walkers


        Returns
        -------
        resampled_walkers : list of resampled_walkers

        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.

        resampler_data :list of dict of str: value
            The resampler records resulting from the resampler actions.

        """

        #initialize the parameters
        num_walkers = len(walkers)
        walker_weights = [walker.weight for walker in walkers]
        num_walker_copies = [1 for i in range(num_walkers)]
        dcd_folder = self.dcd_folder
        cycle_id = cycle_id
        n_bins = n_bins
        max_bins = max_bins
        # calculate  distances
        distance_arr, distance_matrix, images, n_bins = self.get_dist(walkers,folder = dcd_folder,increment = self.increment , n_d = self.n_d, it = cycle_id,n_bins = n_bins,max_bins = max_bins)

        # Closest walker distance
        cw_dist = np.max(images)


        # determine cloning and merging actions to be performed, by
        # maximizing the variation, i.e. the Decider
        resampling_data, variation, happen = self.decide(walker_weights, num_walker_copies, distance_arr, distance_matrix)


        file = open(f'{self.path}/folding/Info_{self.run_id}.txt', 'a')
        file.write(f'Clst walk. FNC: {cw_dist}'+'\t'+f'Resampling happend: {happen}'+'\n')
        file.close()

        # convert the target idxs and decision_id to feature vector arrays
        for record in resampling_data:
            record['target_idxs'] = np.array(record['target_idxs'])
            record['decision_id'] = np.array([record['decision_id']])

        # update trajectory files according to the resampling data
        update_dcd_files(resampling_data, dcd_folder = dcd_folder)

        # actually do the cloning and merging of the walkers
        resampled_walkers = self.DECISION.action(walkers, [resampling_data])

        # flatten the distance matrix and give the number of walkers
        # as well for the resampler data, there is just one per cycle
        resampler_data = [{'distance_array' : distance_arr,
                           'num_walkers' : np.array([len(walkers)]),
                           'variation' : np.array([variation]),
                           'images' : np.ravel(np.array(images)),
                           'image_shape' : np.array(images[0].shape)}]

        return resampled_walkers, resampling_data, resampler_data, n_bins
