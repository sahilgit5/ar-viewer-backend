from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import tempfile
import logging
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import gc

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def estimate_depth(image):
    """Simple depth estimation using edge detection and Gaussian blur"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    # Create depth map from edges
    depth = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    
    return depth

def create_3d_model(image, depth):
    """Create a simple 3D model from image and depth map"""
    try:
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Create grid of points
        y, x = np.mgrid[0:h, 0:w]
        z = depth * 10  # Scale depth for better visualization
        
        # Stack points
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        
        # Set colors
        colors = image.reshape(-1, 3) / 255.0
        
        # Create point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Create mesh using Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        
        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        return mesh
    except Exception as e:
        logger.error(f"Error creating 3D model: {str(e)}")
        raise

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            logger.error("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            logger.error("Invalid file type")
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            
            # Read image
            img = cv2.imread(file_path)
            if img is None:
                logger.error("Failed to read image")
                return jsonify({'error': 'Failed to read image'}), 400
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if too large
            max_dimension = 400
            height, width = img_rgb.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img_rgb = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            # Estimate depth
            depth = estimate_depth(img_rgb)
            
            # Create 3D model
            mesh = create_3d_model(img_rgb, depth)
            
            # Save as GLB
            output_path = os.path.join(temp_dir, 'model.glb')
            o3d.io.write_triangle_mesh(output_path, mesh)
            
            logger.info("Successfully generated 3D model")
            return send_file(output_path, mimetype='model/gltf-binary')
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        gc.collect()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 