from super_gradients.training import models
from super_gradients.common.object_names import Models

yolo_nas_pose = models.get(Models.YOLO_NAS_POSE_L, pretrained_weights="coco_pose")

# prediction = yolo_nas_pose.predict("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
prediction = yolo_nas_pose.predict("C:\_data\project\image\\test.jpg")
prediction.show()