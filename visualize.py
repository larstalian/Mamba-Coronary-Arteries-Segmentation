import mayavi.mlab as mlab
import nibabel as nib

img = nib.load('prediction2.nii.gz').get_fdata()
mlab.contour3d(img)
mlab.show()
