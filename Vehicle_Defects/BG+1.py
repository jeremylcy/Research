from calendar import EPOCH
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
from mrcnn.visualize import random_colors
import matplotlib.pyplot as plt
import tensorflow as tf
import imgaug.augmenters as iaa
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.666
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Root directory of the project
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import warnings 
warnings.filterwarnings("ignore")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "base_model.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
classes = ['BG', 'scratch']
config_name = "damage"
num_classes = len(classes)
steps = 200
val_steps = 400
accuracy = 0.70

json_file = "data.json"
num_epoch = 20

############################################################
#  Config
############################################################

class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = config_name

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = num_classes  # Background + phone,laptop and mobile

    # Number of training steps per epoch
    STEPS_PER_EPOCH = steps
    VALIDATION_STEPS = val_steps

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = accuracy

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        for i in range(0, len(classes)):
            if i != 0:
                self.add_class(config_name, i, classes[i])
                #self.add_class("damage", 2, "dent")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir + "/" + json_file)))
        # print(annotations1)
        
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if "regions" in a]
        annotations = [match for match in annotations if "regions" in match]
        
        # Add images
        for a in annotations:

            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions']]
            names = [r['region_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                config_name,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        # image_info = self.image_info[image_id]
        #if image_info["source"] != "damage":
        #    return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1
            mask[rr, cc, i] = 1

        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        
        #print(class_names[0])
        # "name" is the attributes name decided when labeling, etc. 'region_attributes': {name:'a'}
        for x in range(len(classes)):
            for i, p in enumerate(class_names):
                if classes[x] != "BG":
                    if p['name'] == classes[x]:
                        class_ids[i] = x
        
            """ elif p['name'] == 'dent':
                class_ids[i] = 2
            elif p['name'] == 'headlamp':
                class_ids[i] = 3
            elif p['name'] == 'door':
                class_ids[i] = 4
            elif p['name'] == 'hood':
                class_ids[i] = 5 """
        
        # class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), class_ids.astype(int)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == config_name:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()
    
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(
            0.5,
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
        ),
        
        iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

        iaa.OneOf([
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ]),
        
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    class InferenceConfig(CustomConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    model_inference = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,
        model_inference, dataset_val, calculate_map_at_every_X_epoch=1, verbose=0)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=num_epoch,
                layers='heads',
                custom_callbacks=None,
                augmentation=augmentation)
    evaluate(model_inference, model=model)

def evaluate(model_inference, model=None):
    if model is not None:
        model_train=model
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()
    eval, score = modellib.EvalImage(dataset_val, model_inference, model_train)

    print("Average precision (%) = {0:.2f}, Average accuracy (%) = {1:.2f}".format(eval, score))
    
def gray_scale(image):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = (skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255)
    
    # Copy color pixels from the original color image where mask is set
    splash = gray

    return splash

def detect_and_color_splash(model, image_path=None, video_path=None, f_name=None):
    
    assert image_path or video_path
    colors = random_colors(len(classes))
    class_dict = {name: color for name, color in zip(classes, colors)}
    # Image or video?
    if image_path:
        
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))

        # Read image
        image = skimage.io.imread(image_path)
        
        # print(image)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        
        gray_image = gray_scale(image)
        # print(r)
        # Color splash
        #splash = color_splash(image, r['masks'])
        
        # Save output
        file_name = "splash_" + f_name
        save_file_name = os.path.join(os.getcwd(), file_name)
        
        display_instances(image, r['rois'], r['masks'], r['class_ids'],
                        classes, r['scores'], path=save_file_name, make_image=True, class_dict=class_dict,
                        show_bbox=False)
        # print(r['scores'])
        # plt.savefig(save_file_name)
        # skimage.io.imsave(file_name, plt)

    elif video_path: #not fix yet (in progress)
        if args.video == "0":
            width=1280
            height=720
            vcapture = cv2.VideoCapture(0)
            vcapture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            vcapture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            vcapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'mp4v'))
            vcapture.set(cv2.CAP_PROP_FPS, 60)
            success = True
            while success:

                # Read next image
                success, image = vcapture.read()
                if success:
                    r = model.detect([image], verbose=0)[0]
                    splash = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                classes, r['scores'], make_video=True, class_dict=class_dict)
                    
                    cv2.imshow("splash", splash)
                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break
            
            vcapture.release()
            cv2.destroyAllWindows()
        else:
            # Video capture
            vcapture = cv2.VideoCapture(video_path)
            width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vcapture.get(cv2.CAP_PROP_FPS)

            # Define codec and create video writer
            file_name = "splash_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
            vwriter = cv2.VideoWriter(file_name,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, (width, height))
            vwriter.write(splash)
            
            count = 0
            success = True
            #colors = random_colors(len(class_names))
            # file_path = "/tmp_video_images/"
            
            while success:

                # Read next image
                plt.clf()
                plt.close()
                success, image = vcapture.read()

                if success:
                    if count == 0:
                        print("\n----------------------------------------------- Preparing All Videos -----------------------------------------------\n")        
                    elif count != 0:
                        print("frame: ", count)
                    # OpenCV returns images as BGR, convert to RGB
                    
                    #image1 = cv2.resize(image, dsize=(width,height))
                    #image = image[..., ::-1]
                    gray_image = gray_scale(image)
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]

                    # Color splash
                    # splash = color_splash(image, r['masks'])
                    # save_file_name = os.path.join(dirName, "/" + file_name)
                    save_file_name = os.path.join(os.getcwd(), "temp.png")
                    splash = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                classes, r['scores'], path=save_file_name, make_video=True, class_dict=class_dict)
                    # img = skimage.io.imread(save_file_name)
                    # RGB -> BGR to save image to video
                    #img = img[..., ::-1]
                    # Add image to video writer
                    #print(np.array(splash, dtype=np.uint8))
                vwriter.write(splash)
                count += 1
            #new_path = os.path.join(dirName, '/*.jpg')'''
            '''new_path = os.path.join(os.getcwd(), '*.jpg')
            img_array = []
            for filename in glob.glob(new_path):
                img = cv2.imread(filename)
                img_array.append(img)
            for i in range(0, len(img_array)):
                vwriter.write(img_array[i])'''
            vwriter.release()
            masked_video_path = os.path.join(os.getcwd(), file_name)
            print("Saved to ", masked_video_path)
            #os.rmdir(dirName)
            #for filename in glob.glob(new_path):
            # os.remove(save_file_name)

        # print("Saved to ", file_name)

def all_files_in_folder(model, image_path=None, video_path=None):
    assert image_path or video_path
    if image_path:
        print("#####################################################################################################################")
        print("----------------------------------------------- Processing All Images -----------------------------------------------")
        print("#####################################################################################################################")
        print("\n----------------------------------------------- Preparing All Images -----------------------------------------------\n")
        folder_name = "results_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
        folder_path = os.path.join(os.getcwd(), "Results")

        for i, file in enumerate(os.listdir(args.image_path)):
            print("Image number:" + str(i + 1)) 
            detect_and_color_splash(model, image_path=os.path.join(image_path, file), f_name=str(file))
    elif video_path:
        print("#####################################################################################################################")
        print("----------------------------------------------- Processing All Videos -----------------------------------------------")
        print("#####################################################################################################################")
        if args.video == "0":
            detect_and_color_splash(model, video_path="0")
        else:
            for i, file in enumerate(os.listdir(args.video)):
                print("Video number:" + str(i + 1))
                detect_and_color_splash(model, video_path=os.path.join(video_path, file)) 

############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash' or 'evaluate")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image_path', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video_path', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for Training model on both train and val datasets"
    elif args.command == "evaluate":
        assert args.dataset, "Argument --dataset is required for Testing model on val dataset"
    elif args.command == "splash":
        assert args.image_path or args.video_path,\
               "Provide --image_path or --video_path to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):

            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    
    if args.weights.lower() == "coco":

        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        # detect_and_color_splash(model, image_path=args.image, video_path=args.video)
        all_files_in_folder(model, image_path=args.image_path, video_path=args.video_path)
    elif args.command == "evaluate":
        evaluate(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash' or 'evaluate'".format(args.command))
	