import gradio as gr


def yolov9_inference(model_path, device, conf_threshold, iou_threshold, img_path, size=640):
    """
    Load a YOLOv9 model, configure it, perform inference on an image, and optionally adjust 
    the input size and apply test time augmentation.
    
    :param model_path: Path to the YOLOv9 model file.
    :param device: Computation device, 'cpu' or 'cuda'.
    :param conf_threshold: Confidence threshold for NMS.
    :param iou_threshold: IoU threshold for NMS.
    :param img_path: Path to the image file.
    :param size: Optional, input size for inference.
    :return: A tuple containing the detections (boxes, scores, categories) and the results object for further actions like displaying.
    """
    # Import YOLOv9
    import yolov9
    
    # Load the model
    model = yolov9.load(model_path, device=device)
    
    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    
    # Perform inference
    results = model(img_path, size=size)

    # Optionally, show detection bounding boxes on image
    save_path = 'output/'
    results.save(labels=True, save_dir=save_path)
    

    return save_path + 'elon.jpg'


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
        value="gelan-e.pt",
    ),
    gr.Slider(minimum=320, maximum=1280, value=1280, step=32, label="Image Size"),
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