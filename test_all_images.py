import cv2
from insightface.app import FaceAnalysis
from pathlib import Path

# Initialize the app
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Get all images
img_dir = Path('python-package/insightface/data/images')
images = sorted(img_dir.glob('*.[jp][pn]g'))

# Process each image
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'FAIL: Cannot read {img_path}')
        continue
    
    faces = app.get(img)
    print(f'{img_path.name}: {len(faces)} faces detected')
    
    # Draw results
    output_path = img_path.with_stem(img_path.stem + '_out')
    cv2.imwrite(str(output_path), app.draw_on(img, faces))
    print(f'  -> Saved to {output_path.name}')