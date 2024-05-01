from gradio_client import Client
import gradio_client as gr
import os 
import time
import re
import sys

style_template = {"style1": "Film Noir", "style2": "Vibrant Color","style3": "Snow",
                  "style4": "Neon","style5": "Jungle", "style6": "Mars", "style7": "Spring Festival", 
                  "style8": "Watercolor", "style9": "Line art"}
negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green"

client = Client("https://instantx-instantid.hf.space/",
                ssl_verify=False
                )


def process_image(face_image_path, prompt, style):
    face_image = gr.file(face_image_path)
    result = client.predict(
			face_image,	# filepath  in 'Upload a photo of your face' Image component
			None,	# filepath  in 'Upload a reference pose image (Optional)' Image component
			prompt,	# str  in 'Prompt' Textbox component
			negative_prompt,	# str  in 'Negative Prompt' Textbox component
			style,	# Literal['(No style)', 'Spring Festival', 'Watercolor', 'Film Noir', 'Neon', 'Jungle', 'Mars', 'Vibrant Color', 'Snow', 'Line art']  in 'Style template' Dropdown component
			20,	# float (numeric value between 1 and 100) in 'Number of sample steps' Slider component
			0.8,	# float (numeric value between 0 and 1.5) in 'IdentityNet strength (for fidelity)' Slider component
			0.8,	# float (numeric value between 0 and 1.5) in 'Image adapter strength (for detail)' Slider component
			0.4,	# float (numeric value between 0 and 1.5) in 'Pose strength' Slider component
			0.4,	# float (numeric value between 0 and 1.5) in 'Canny strength' Slider component
			0.4,	# float (numeric value between 0 and 1.5) in 'Depth strength' Slider component
			["pose"],	# List[Literal['pose', 'canny', 'depth']]  in 'Controlnet' Checkboxgroup component
			5,	# float (numeric value between 0.1 and 20.0) in 'Guidance scale' Slider component
			42,	# float (numeric value between 0 and 2147483647) in 'Seed' Slider component
			"EulerDiscreteScheduler",	# Literal['DEISMultistepScheduler', 'HeunDiscreteScheduler', 'EulerDiscreteScheduler', 'DPMSolverMultistepScheduler', 'DPMSolverMultistepScheduler-Karras', 'DPMSolverMultistepScheduler-Karras-SDE']  in 'Schedulers' Dropdown component
			False,	# bool  in 'Enable Fast Inference with LCM' Checkbox component
			True,	# bool  in 'Enhance non-face region' Checkbox component
			api_name="/generate_image"
	)
    return result

if __name__ == '__main__':
    base_images_folder = './input/cropped/'
    
    # Read from system arguments or set default.
    style_list = sys.argv[1].split(",") if len(sys.argv) > 1 else ["style1"]
    print(f'Style list: {style_list}')
    
    prompts = os.listdir(base_images_folder)
    print(f'Prompts: {prompts}')

    for prompt in prompts:
        images_folder = os.path.join(base_images_folder, prompt)
        if not os.path.isdir(images_folder):
            continue
        
        for selected_style in style_list:
            print(f'Running pipeline for {selected_style} with prompt "{prompt}"')
            output_path = f'./output/images/{selected_style}/'
            # Create output folder if not exists
            os.makedirs(output_path, exist_ok=True)

            list_of_images = os.listdir(images_folder)
            list_of_images = [img for img in list_of_images if img.endswith('.jpg')]
            list_of_images = sorted(list_of_images)

            for image_name in list_of_images:
                output_image_path = os.path.join(output_path, image_name)
                
                if os.path.exists(output_image_path):
                    print(f"Skipping image: {image_name}")
                    continue
                
                retry_count = 0
                success = False

                while not success and retry_count < 3:  # Retry up to 3 times
                    try:
                        print(f"Processing image: {image_name}")
                        face_image_path = os.path.join(images_folder, image_name)
                        result = process_image(face_image_path, prompt, style_template[selected_style])
                        os.rename(result[0], output_image_path)
                        success = True  # Set success to True if process_image completes without error
                    except Exception as e:
                        client.reset_session()
                        retry_count += 1
                        print(f"Error processing image: {image_name}. Error: {e}")
                        # Extract the retry wait time from the error message
                        match = re.search(r"Please retry in (\d+):(\d+):(\d+)", str(e))
                        if match:
                            hours, minutes, seconds = map(int, match.groups())
                            wait_time = hours * 3600 + minutes * 60 + seconds
                            buffer_time = 10  # Adding a buffer of 10 seconds
                            total_wait_time = wait_time + buffer_time
                            print(f"Attempt {retry_count}: Retrying in {total_wait_time // 60} minutes and {total_wait_time % 60} seconds...")
                            time.sleep(total_wait_time)  # Wait the total wait time before retrying
                        else:
                            print(f"Attempt {retry_count}: Retrying in 5 minutes...")
                            time.sleep(300)  # Default wait time if parsing fails
                time.sleep(5)
            
            # List the number of styled images generated
            generated_images = os.listdir(output_path)
            generated_images = [img for img in generated_images if img.endswith('.jpg')]
            print(f"Total {len(generated_images)} {selected_style} images generated for prompt {prompt}.")
