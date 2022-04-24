# ACM FAQ

## Is it possible to use an alternative marker configuration?

Although the marker positions shown in our publication were carefully selected for their connection to their respective joint, it is in principle possible to choose alternative marker configuration. For this, the following steps have to be performed:

1. Prepare a modified models.py, where
joint_marker_order gives a list of the names of the new set of markers and
joint_marker_index links each of these new markers to a joint.

*Note that it is important to be consistent with the naming of the new joints & markers (i.e. for joints: 'joint\_\*' & for surface markers: 'spot\_\*' in labeling GUI, otherwise 'marker\_\*\_start'). Particularly, function 'initialize\_x' in calibration.py currently assumes that the model contains markers with the names 'spot\_head\_002' and 'spot\_head\_003' or 'spot_spine\_\*' and/or 'spot_tail\_\*' (this function is used to get very rough initial pose alignments in calibration.py and initialization.py).*
2. Modify anatomy.py for new constraints according to the methods section
"Constraining surface marker positions based on body symmetry"

*This concerns the functions 'get_bounds_markers', 'get_bounds_pose' and 'get_skeleton_coords0'*