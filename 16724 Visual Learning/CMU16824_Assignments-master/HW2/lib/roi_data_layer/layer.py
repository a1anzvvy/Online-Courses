# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch,get_weak_minibatch
import numpy as np
import yaml
import time, math, sys
from multiprocessing import Process, Queue
from utils.cython_bbox import bbox_overlaps

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass




class RoIWeakDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()   # since batch is 1, only one integer here
            minibatch_db = [self._roidb[i] for i in db_inds]    # one entry in roidb

# [{'gt_classes': array([15,  2], dtype=int32), 'max_classes': array([15, 15, 15, ..., 15,  2,  2]), 'image': '/home/xinshuo/Dropbox/hw/16824/HW2/CMU16824hw2/data/VOCdevkit2007/VOC2007/JPEGImages/004622.jpg', 'boxscores': array([ 0.25462374,  0.23050041,  0.22954161, ...,  0.01787845,
#         0.01787303,  0.01787046], dtype=float32), 'bbox_targets': array([[ 0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.],
#        ..., 
#        [ 0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  0.,  0.]], dtype=float32), 'flipped': False, 'width': 375, 'boxes': array([[  0,  57, 374, 499],
#        [190,  83, 374, 499],
#        [  0,   0, 374, 402],
#        ..., 
#        [330, 414, 370, 435],
#        [211, 366, 271, 388],
#        [288, 292, 323, 323]], dtype=int16), 'max_overlaps': array([ 0.89258087,  0.5202657 ,  0.62884766, ...,  0.00608309,
#         0.01089954,  0.00894959], dtype=float32), 'height': 500, 'seg_areas': array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32), 'gt_overlaps': <3834x21 sparse matrix of type '<type 'numpy.float32'>'
#         with 3774 stored elements in Compressed Sparse Row format>}]

            return get_weak_minibatch(minibatch_db, self._num_classes)

    # useless right now
    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,       # 1, 600, 1000
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

        # if cfg.DEBUG:
        #     print(cfg.TRAIN.IMS_PER_BATCH)
        #     print(max(cfg.TRAIN.SCALES))
        #     print(cfg.TRAIN.MAX_SIZE)
        #     time.sleep(1000)

        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1,20)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1
        if cfg.TRAIN.BOXSCORE:
            top[idx].reshape(1)
            self._name_to_top_map['boxscores'] = idx
            idx += 1
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map

        # if cfg.DEBUG:
        #     print(idx)
        #     print(len(self._name_to_top_map))
        #     print(len(top))

        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        # if cfg.DEBUG:
            # print(blobs)
            # time.sleep(1000)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)




#####TODO: Implement any python layers here #########

# prediction 1x20 with label 1x20
class BinaryLossLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        assert(len(bottom)==2)
        assert(len(top)==1)
    
    # loss = log(y * (y_pred - 0.5) + 0.5)
    # episolon is for perturbing the data to avoid math domain error
    def forward(self, bottom, top):
        loss = 0
        for i in xrange(20):
            if bottom[0].data[0, i] > 1:
                bottom[0].data[0, i] = 1;
            if bottom[0].data[0, i] < 0:
                bottom[0].data[0, i] = 0;                 
            assert bottom[1].data[0, i] == 1 or bottom[1].data[0, i] == -1, 'The binary label is not as expected, which is %f' % bottom[1].data[0, i]
            loss += math.log((bottom[0].data[0, i] - 0.5) * bottom[1].data[0, i] + 0.5 + sys.float_info.epsilon)
        top[0].data[...] = -loss

    # diff = y / (y * (y_pred - 0.5) + 0.5)
    def backward(self, top, propagate_down, bottom):
        for i in xrange(20):
            if bottom[0].data[0, i] > 1:
                bottom[0].data[0, i] = 1;
            if bottom[0].data[0, i] < 0:
                bottom[0].data[0, i] = 0;  
            bottom[0].diff[0, i] = -1 * (bottom[1].data[0, i] / (bottom[1].data[0, i] * (bottom[0].data[0, i] - 0.5) + 0.5  + sys.float_info.epsilon))

    def reshape(self, bottom, top):
        assert len(bottom[0].shape) == 2, 'The prediction shape is not correct'
        assert bottom[0].shape[0] == 1, 'The prediction shape is not correct'
        assert bottom[0].shape[1] == 20, 'The prediction shape is not correct'
        assert len(bottom[1].shape) == 2, 'The label shape is not correct'
        assert bottom[1].shape[0] == 1, 'The label shape is not correct'
        assert bottom[1].shape[1] == 20, 'The label shape is not correct'
        # self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)


#####################################################


# from Nx20 to 1x20xN
class SecretAssignmentLayer(caffe.Layer):
	
    def setup(self, bottom, top):
        assert(len(bottom)==1)
        assert(len(top)==1)
    
    def forward(self, bottom, top):
        top[0].data[0,:] = np.swapaxes(bottom[0].data[:],0,1)
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[:] = np.swapaxes(top[0].diff[0,:,:],0,1)

    def reshape(self, bottom, top):
        top[0].reshape(1, bottom[0].data.shape[1], bottom[0].data.shape[0])
