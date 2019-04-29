from lu_vp_detect.vp_detection import vp_detection
import cv2
import json
import numpy as np
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_vp_axes_image(img_dir, principal_point, vps_2D,  show_image=False):
	"""
	Once the VP detection algorithm runs, show the axes that the vps point to

	Args:
		show_image: Show the image in an OpenCV imshow window
					(default=false)

	Returns:
		The image with axes

	Raises:
		ValueError: If the path to the image is not a string or None
	"""

	img = cv2.imread(img_dir)
	if len(img.shape) == 2:  # If grayscale, artificially make into RGB
		img = np.dstack([img, img, img])

	# Reset principal point if we haven't set it yet
	if principal_point is None:
		rows, cols = img.shape[:2]
		principal_point = np.array([cols / 2.0, rows / 2.0],
										 dtype=np.float32)

	colours = 255 * np.eye(3)
	# BGR format
	# First row is red, second green, third blue
	colours = colours[:, ::-1].astype(np.int).tolist()

	# For each cluster of lines, draw them in their right colour
	for i in range(3):
		cv2.line(img, (principal_point[0], principal_point[1]), (vps_2D[i][0], vps_2D[i][1]),
				 colours[i], 2, cv2.LINE_AA)

	# Show image if necessary
	if show_image:
		cv2.imshow('VP Debug Image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return img

def main():
    input_path = "./hard.jpg"
    #input_path = "test_image.jpg"

    # find out which class the image belong to {"one_star", "two_star"}
    is_one_star = True

    length_thresh = 25
    principal_point = None
    focal_length = 1384
    debug_mode = True
    debug_show = False
    debug_save = True
    debug_path = None
    seed = 1337
    print('Input path: {}'.format(input_path))
    print('Seed: {}'.format(seed))
    print('Line length threshold: {}'.format(length_thresh))
    print('Focal length: {}'.format(focal_length))

    # Create object
    vpd = vp_detection(length_thresh, principal_point, focal_length, seed, is_one_star)

    # Run VP detection algorithm, get all of the 3D vps
    vps = vpd.find_vps(input_path)
    print('Principal point: {}'.format(vpd.principal_point))

    # Show VP information
    print("The vanishing points in 3D space are: ")
    for i, vp in enumerate(vps):
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    #
    vp2D = vpd.vps_2D
    print("\nThe vanishing points in image coordinates are: ")
    for i, vp in enumerate(vp2D):
        print("Vanishing Point {:d}: {}".format(i + 1, vp))

    # Extra stuff
    if debug_mode or debug_show:
        # save line clustering img
        img_save = vpd.create_debug_VP_image(debug_show, debug_path)
        if debug_save:
            print("save img with lines")
            img_save_dir = input_path.split('jpg')[0] + '_lines.jpg'
            cv2.imwrite(img_save_dir, img_save)

        # save axes img
        img_axes = create_vp_axes_image(input_path, principal_point, vp2D, debug_show)
        if debug_save:
            img_axes_save_dir = input_path.split('jpg')[0] + '_axes_save.jpg'
            cv2.imwrite(img_axes_save_dir, img_axes)
if __name__ == "__main__":
	main()