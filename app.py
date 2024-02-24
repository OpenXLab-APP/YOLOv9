import gradio as gr
import spaces
import os
from huggingface_hub import hf_hub_download


def download_models(model_id):
    hf_hub_download("merve/yolov9", filename=f"{model_id}", local_dir=f"./{model_id}")
    return f"./{model_id}"


@spaces.GPU
def yolov9_inference(img_path, model_id, image_size, conf_threshold, iou_threshold):
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
    model_path = download_models(model_id)
    model = yolov9.load(model_path, device="cuda")
    
    # Set model parameters
    model.conf = conf_threshold
    model.iou = iou_threshold
    
    # Perform inference
    results = model(img_path, size=image_size)

    # Optionally, show detection bounding boxes on image
    output = results.render()
    
    return output[0]


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                img_path = gr.Image(type="filepath", label="Image")
                model_path = gr.Dropdown(
                    label="Model",
                    choices=[
                        "gelan-c.pt",
                        "gelan-e.pt",
                        "yolov9-c.pt",
                        "yolov9-e.pt",
                    ],
                    value="gelan-e.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.5,
                )
                yolov9_infer = gr.Button(value="Inference")

            with gr.Column():
                output_numpy = gr.Image(type="numpy",label="Output")

        yolov9_infer.click(
            fn=yolov9_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
        )
        
        gr.Examples(
            examples=[
                [
                    "zidane.jpg",
                    "gelan-e.pt",
                    640,
                    0.4,
                    0.5,
                ],
            ],
            fn=yolov9_inference,
            inputs=[
                img_path,
                model_path,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_numpy],
            cache_examples=True,
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Follow me for more!
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>  | <a href='https://www.huggingface.co/kadirnar/' target='_blank'>HuggingFace</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)