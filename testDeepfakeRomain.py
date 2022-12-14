import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings


warnings.filterwarnings("ignore")

#Image source du visage à utiliser
source_image = imageio.imread("C:/Users/caban/Pictures/deepfake.png")

#Video dans laquelle on veut remplacer le visage
reader = imageio.get_reader("C:/Users/caban/Downloads/Veran.mp4")

# Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]


def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


HTML(display(source_image, driving_video).to_html5_video())

from demo import load_checkpoints

generator, kp_detector = load_checkpoints(config_path='/content/first-order-model/config/vox-256.yaml',
                                          checkpoint_path='/content/vox-adv-cpk.pth.tar')

# Perform image animation

from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

# save resulting video
imageio.mimsave('../generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
# video can be downloaded from /content folder

# HTML(display(source_image, driving_video, predictions).to_html5_video())

# In the cell above we use relative keypoint displacement to animate the objects. We can use absolute coordinates instead,
# but in this way all the object proporions will be inherited from the driving video.
# For example Putin haircut will be extended to match Trump haircut.

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=False,
                             adapt_movement_scale=True)
HTML(display(source_image, driving_video, predictions).to_html5_video())

# Running on your data

source_image = imageio.imread('/content/' + Nom_Fichier_Image)
driving_video = imageio.mimread('/content/' + Nom_Fichier_Video, memtest=False)

# Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True,
                             adapt_movement_scale=True)

imageio.mimsave('generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)

HTML(display(source_image, driving_video, predictions).to_html5_video())