from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
import tempfile
import shutil
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import gc
import psutil
import sys

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory limit (adjust based on your Render instance)
MAX_MEMORY_PERCENT = 80

def check_memory():
    memory = psutil.virtual_memory()
    if memory.percent > MAX_MEMORY_PERCENT:
        logger.warning(f"High memory usage detected: {memory.percent}%")
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models with error handling and lazy loading
yolo_model = None
dpt_processor = None
dpt_model = None

def initialize_models():
    global yolo_model, dpt_processor, dpt_model
    try:
        logger.info("Initializing models...")
        check_memory()
        
        if yolo_model is None:
            # Use a smaller YOLO model
            yolo_model = YOLO('yolov8n.pt')
            check_memory()
        
        if dpt_processor is None:
            # Use the correct model name
            dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            check_memory()
        
        if dpt_model is None:
            dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            check_memory()
            
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        # Don't raise the error, just log it
        logger.error("Application will continue without depth estimation model")
        # Set models to None to prevent usage
        dpt_processor = None
        dpt_model = None

# Initialize models when the application starts
initialize_models()

def process_image(image_path):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image if too large
        max_dimension = 400  # Reduced from 800
        height, width = img_rgb.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img_rgb = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # YOLOv8 object detection
        results = yolo_model(img_rgb)
        if len(results[0].boxes) == 0:
            logger.warning(f"No objects detected in image: {image_path}")
            return None
        
        # Get the largest detected object
        boxes = results[0].boxes
        largest_box = max(boxes, key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        
        # Add padding to the crop
        padding = 10  # Reduced from 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_rgb.shape[1], x2 + padding)
        y2 = min(img_rgb.shape[0], y2 + padding)
        
        # Crop the image
        cropped_img = img_rgb[y1:y2, x1:x2]
        
        # Check if depth estimation models are available
        if dpt_processor is None or dpt_model is None:
            logger.warning("Depth estimation models not available, using simple depth estimation")
            # Create a simple depth map (placeholder)
            depth = np.zeros(cropped_img.shape[:2], dtype=np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        else:
            # Depth estimation
            inputs = dpt_processor(images=Image.fromarray(cropped_img), return_tensors="pt")
            with torch.no_grad():
                outputs = dpt_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Normalize depth
            predicted_depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=cropped_img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth = predicted_depth.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            
            # Clear memory
            del inputs, outputs, predicted_depth
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            gc.collect()
        
        return cropped_img, depth
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def reconstruct_3d(images, depths):
    try:
        # Create point cloud using Open3D
        pcd = o3d.geometry.PointCloud()
        
        # Combine points from all images
        points = []
        colors = []
        
        for img, depth in zip(images, depths):
            h, w = img.shape[:2]
            y, x = np.mgrid[0:h, 0:w]
            z = depth
            
            # Convert to 3D points
            points_3d = np.stack([x, y, z], axis=-1)
            points.append(points_3d.reshape(-1, 3))
            colors.append(img.reshape(-1, 3))
        
        points = np.vstack(points)
        colors = np.vstack(colors)
        
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Remove outliers with more aggressive parameters
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Poisson surface reconstruction with reduced parameters
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Clear memory
        del points, colors, pcd
        gc.collect()
        
        return mesh
    except Exception as e:
        logger.error(f"Error in 3D reconstruction: {str(e)}")
        raise

@app.route('/process', methods=['POST'])
def process_images():
    try:
        check_memory()
        if 'images' not in request.files:
            logger.error("No images provided in request")
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files:
            logger.error("Empty files list")
            return jsonify({'error': 'No images selected'}), 400
        
        # Limit number of images
        if len(files) > 3:
            return jsonify({'error': 'Maximum 3 images allowed'}), 400
        
        logger.info(f"Processing {len(files)} images")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_images = []
            processed_depths = []
            
            # Process each image
            for file in files:
                check_memory()
                if file.filename == '':
                    continue
                    
                if not allowed_file(file.filename):
                    logger.warning(f"Invalid file type: {file.filename}")
                    continue
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                # Process image
                result = process_image(file_path)
                if result is None:
                    continue
                    
                processed_img, processed_depth = result
                processed_images.append(processed_img)
                processed_depths.append(processed_depth)
                
                # Clear memory after each image
                check_memory()
            
            if not processed_images:
                logger.error("No valid images processed")
                return jsonify({'error': 'No valid images processed'}), 400
            
            logger.info(f"Successfully processed {len(processed_images)} images")
            
            # Reconstruct 3D model
            check_memory()
            mesh = reconstruct_3d(processed_images, processed_depths)
            
            # Save as GLB
            output_path = os.path.join(temp_dir, 'model.glb')
            o3d.io.write_triangle_mesh(output_path, mesh)
            
            logger.info("Successfully generated 3D model")
            return send_file(output_path, mimetype='model/gltf-binary')
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        check_memory()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 