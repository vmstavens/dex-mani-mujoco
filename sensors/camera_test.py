import mujoco_py
import cv2
import numpy as np
# from pcl import PointCloud, PointXYZ, PointXYZRGB, visualization

class RGBD_Mujoco:
    def __init__(self):
        self.color_buffer = None
        self.depth_buffer = None
        self.extent = None
        self.z_near = None
        self.z_far = None
        self.f = None
        self.cx = None
        self.cy = None
        self.color_image = None
        self.depth_image = None

    def linearize_depth(self, depth):
        depth_img = self.z_near * self.z_far * self.extent / (self.z_far - depth * (self.z_far - self.z_near))
        return depth_img

    def set_camera_intrinsics(self, model, camera, viewport):
        fovy = model.cam_fovy[camera.fixedcamid] / 180 * np.pi / 2
        self.f = viewport.height / (2 * np.tan(fovy))
        self.cx = viewport.width / 2
        self.cy = viewport.height / 2

    def get_RGBD_buffer(self, model, viewport, context):
        self.color_buffer = np.zeros((viewport.height * viewport.width * 3,), dtype=np.uint8)
        self.depth_buffer = np.zeros((viewport.height * viewport.width,), dtype=np.float32)
        context.read_pixels(self.color_buffer, self.depth_buffer, viewport)
        
        self.extent = model.stat.extent
        self.z_near = model.vis.map.znear
        self.z_far = model.vis.map.zfar

        img_size = (viewport.width, viewport.height)
        self.color_image = cv2.cvtColor(self.color_buffer.reshape((img_size[1], img_size[0], 3), order='C'), cv2.COLOR_BGR2RGB)
        self.color_image = np.flipud(self.color_image)
        
        self.depth_image = self.linearize_depth(np.flipud(self.depth_buffer).reshape(img_size))

    # def generate_pointcloud(self):
    #     cloud = PointCloud()
    #     for i in range(self.depth_image.shape[0]):
    #         for j in range(self.depth_image.shape[1]):
    #             depth = self.depth_image[i, j]
    #             if depth < self.z_far:
    #                 point = PointXYZ()
    #                 point.x = (j - self.cx) * depth / self.f
    #                 point.y = (i - self.cy) * depth / self.f
    #                 point.z = depth
    #                 cloud.points.append(point)
    #     return cloud

    # def generate_color_pointcloud(self):
    #     assert self.color_image.shape == self.depth_image.shape
    #     rgb_cloud = PointCloud()
    #     for i in range(self.color_image.shape[0]):
    #         for j in range(self.color_image.shape[1]):
    #             depth = self.depth_image[i, j]
    #             if depth < self.z_far:
    #                 rgb_point = PointXYZRGB()
    #                 rgb_point.x = (j - self.cx) * depth / self.f
    #                 rgb_point.y = (i - self.cy) * depth / self.f
    #                 rgb_point.z = depth
    #                 bgr_pixel = self.color_image[i, j, ::-1]  # Convert RGB to BGR
    #                 rgb_point.b = bgr_pixel[0]
    #                 rgb_point.g = bgr_pixel[1]
    #                 rgb_point.r = bgr_pixel[2]
    #                 rgb_cloud.points.append(rgb_point)
    #     return rgb_cloud

# Usage example
if __name__ == "__main__":

    rgb_mujoco = RGBD_Mujoco()
    rgb_mujoco.set_camera_intrinsics(mujoco_env.model, mj_viewer.cam, mj_viewer.vopt.rect)
    rgb_mujoco.get_RGBD_buffer(mujoco_env.model, mj_viewer.vopt.rect, mj_viewer._get_viewer_context())