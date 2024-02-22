import gradio as gr


def yolov9_inference(img_path, model_path,image_size, conf_threshold, iou_threshold):
    """
    Load a YOLOv9 model, configure it, perform inference on an image, and optionally adjust 
    the input size and apply test time augmentation.
    
    :param model_path: Path to the YOLOv9 model file.
    :param conf_threshold: Confidence threshold for NMS.
    :param iou_threshold: IoU threshold for NMS.
    :param img_path: Path to the image file.
    :param size: Optional, input size for inference.
    :return: A tuple containing the detections (boxes, scores, categories) and the results object for further actions like displaying.
    """
    # Import YOLOv9
    import yolov9
    
    # Load the model
    model = yolov9.load(model_path, device="cpu")
    
    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    
    # Perform inference
    results = model(img_path, size=image_size)

    # Optionally, show detection bounding boxes on image
    save_path = 'output/'
    results.save(labels=True, save_dir=save_path, exist_ok=True)
    
    output_path = save_path + img_path
    print(f"Output image saved to {output_path}")
    return output_path


inputs = [
    gr.Image(label="Input Image"),
    gr.Dropdown(
        label="Model",
        choices=[
            "gelan-c.pt",
            "gelan-e.pt",
            "yolov9-c.pt",
            "yolov9-e.pt",
        ],
        value="gelan-c.pt",
    ),
    gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")
title = "YOLOv9"

demo_app = gr.Interface(
    fn=yolov9_inference,
    inputs=inputs,
    outputs=outputs,
    title=title,
)
demo_app.launch(debug=True)