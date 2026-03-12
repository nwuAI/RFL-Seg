import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from PIL import Image
import matplotlib.pyplot as plt


class Config:
    """Configuration class to manage model parameters, paths, and prompt templates"""

    def __init__(self):
        self.device = "cuda"
        self.best_model_path = None
        self.img_root = "./USData/Breast-BUSI/"
        self.mask_root = "./output/BUSI/deeplabv3+_resnet50/coarse_mask/"
        self.output_dir = "./output/BUSI/deeplabv3+_resnet50/SD15_infer/coarse_gen"
        self.is_show_diff = False
        self.modality = "ultrasound"  # Required parameter
        self.part = "breast"  # Optional parameter(breast, thyroid)
        self.target_tissue = "breast"  # Optional parameter(breast, thyroid)

        # Prompt template library - categorized by modality
        self.prompt_templates = {
            "ultrasound": [
                "A high-resolution {modality} image of healthy {target_tissue} tissue, with clear anatomical "
                "structures, normal echogenicity (low to medium), smooth tissue boundaries, no abnormal "
                "shadows or lesions, realistic vascular patterns, and natural grayscale distribution.",

                "A detailed {modality} scan showing normal {target_tissue} anatomy with homogeneous echotexture, "
                "well-defined borders, and physiological vascular flow patterns. No signs of pathology."
            ],
            "ct": [
                "A contrast-enhanced {modality} scan of the {part}, depicting normal {target_tissue} parenchyma "
                "with appropriate attenuation values, regular morphology, and no evidence of masses or effusions.",

                "Axial {modality} image demonstrating healthy {target_tissue} tissue with normal density, "
                "smooth contours, and intact anatomical relationships. No pathological findings."
            ],
            "mri": [
                "{modality} scan (T1-weighted) of the {part}, revealing normal {target_tissue} signal intensity, "
                "well-defined tissue planes, and no abnormal enhancement or structural abnormalities.",

                "{modality} (T2-weighted) showing homogeneous {target_tissue} tissue with physiological "
                "signal characteristics, intact architecture, and no signs of inflammation or lesions."
            ]
        }

        # Negative prompt template library - categorized by modality
        self.negative_prompt_templates = {
            "ultrasound": [
                "blurry, low resolution, artifacts, shadows, abnormal echogenicity, irregular "
                "boundaries, lesions, tumors, cysts, fibrosis, pneumonia, pleural effusion, "
                "or any pathological features.",

                "poor image quality, noise, artifacts, heterogeneous echotexture, focal lesions, "
                "abnormal vascularity, pleural thickening, or signs of infection."
            ],
            "ct": [
                "motion artifacts, streaking, low contrast, abnormal densities, masses, nodules, "
                "effusions, hemorrhage, edema, or any pathological processes.",

                "artifacts, beam hardening, metal streaks, focal lesions, infiltrates, masses, "
                "lymphadenopathy, or signs of trauma."
            ],
            "mri": [
                "motion artifacts, susceptibility artifacts, low signal-to-noise ratio, abnormal "
                "enhancement, masses, lesions, edema, hemorrhage, or any pathological conditions.",

                "flow voids, chemical shift artifacts, poor tissue contrast, focal signal abnormalities, "
                "masses, inflammation, demyelination, or structural deformities."
            ]
        }

        # Default to use the first template
        self.prompt_template_index = 0
        self.negative_prompt_template_index = 0

    def get_prompt(self) -> str:
        """Generate prompt based on configuration"""
        # Ensure modality exists in template library
        if self.modality not in self.prompt_templates:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # Get templates for the corresponding modality
        templates = self.prompt_templates[self.modality]
        template = templates[self.prompt_template_index % len(templates)]

        # Replace placeholders in template with configuration parameters
        params = {
            "modality": self.modality,
            "part": self.part if self.part else "",
            "target_tissue": self.target_tissue if self.target_tissue else ""
        }

        # Ensure optional parameters exist in template
        if not params["part"]:
            template = template.replace("{part}", "")
        if not params["target_tissue"]:
            template = template.replace("{target_tissue}", "")

        return template.format(**params)

    def get_negative_prompt(self) -> str:
        """Generate negative prompt based on configuration"""
        # Ensure modality exists in template library
        if self.modality not in self.negative_prompt_templates:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # Get templates for the corresponding modality
        templates = self.negative_prompt_templates[self.modality]
        template = templates[self.negative_prompt_template_index % len(templates)]

        # Replace placeholders in template with configuration parameters
        params = {
            "modality": self.modality,
            "part": self.part if self.part else "",
            "target_tissue": self.target_tissue if self.target_tissue else ""
        }

        # Ensure optional parameters exist in template
        if not params["part"]:
            template = template.replace("{part}", "")
        if not params["target_tissue"]:
            template = template.replace("{target_tissue}", "")

        return template.format(**params)

    def set_prompt_template_index(self, index: int) -> None:
        """Set the index of the prompt template"""
        self.prompt_template_index = index

    def set_negative_prompt_template_index(self, index: int) -> None:
        """Set the index of the negative prompt template"""
        self.negative_prompt_template_index = index


class PseudoHealthGenerator:
    """Class to generate pseudo-healthy medical images using Stable Diffusion ControlNet"""

    def __init__(self, config: Config):
        self.device = config.device
        self.best_model_path = config.best_model_path

        # Initialize ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16
        ).to(self.device)

        # Initialize main model (using specialized inpainting pipeline)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            vae=self.vae,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(self.device)

        # Load best model if specified path exists
        if self.best_model_path:
            try:
                checkpoint = torch.load(self.best_model_path, map_location=self.device)
                self.pipe.unet.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded best model: {self.best_model_path}")
            except Exception as e:
                print(f"Error loading best model: {e}, will use pretrained model")

        # Configure optimization parameters
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        # Conditionally enable xformers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not available, falling back to default attention mechanism")

    def prepare_control_image(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Create control image that meets ControlNet requirements"""
        # Ensure image has three channels
        image = self._ensure_three_channels(image)
        # Generate white mask image (RGB format)
        mask_image = np.array(mask)
        mask_image = np.stack([mask_image] * 3, axis=-1) * 255

        # Create control image: original image*0.5 + white mask*0.5
        control_image = cv2.addWeighted(np.array(image), 0.4, mask_image, 0.6, 0)
        return Image.fromarray(control_image)

    def prepare_control_image2(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Create enhanced control image, focusing on guiding ControlNet to generate new structures within the mask"""
        # Ensure image has three channels
        image = self._ensure_three_channels(image)
        # Convert to numpy array
        image_array = np.array(image.convert("RGB"))
        mask_array = np.array(mask.convert("L"))  # Single channel mask

        # Generate healthy tissue texture template
        healthy_template = cv2.GaussianBlur(image_array, (51, 51), sigmaX=15)
        healthy_template[mask_array == 0] = 0  # Apply texture template only in mask area

        # Enhance structure in masked area
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply Sobel edge enhancement only in masked area
        enhanced_mask_edges = np.zeros_like(sobel_magnitude)
        enhanced_mask_edges[mask_array > 0] = sobel_magnitude[mask_array > 0]

        # Build final control image
        control_image = image_array.copy()
        control_image[mask_array > 0] = (
                0.7 * healthy_template[mask_array > 0] +
                0.3 * enhanced_mask_edges[:, :, np.newaxis][mask_array > 0]
        ).astype(np.uint8)

        return Image.fromarray(control_image)

    def _ensure_three_channels(self, image: Image.Image) -> Image.Image:
        """Ensure image has three channels"""
        if len(np.array(image).shape) == 2:
            image = Image.fromarray(np.stack([np.array(image)] * 3, axis=-1))
        return image

    def generate(self, image_path: str, mask_path: str, config: Config, strength: float = 0.95) -> Image.Image:
        """
        Generate pseudo-healthy medical image

        Args:
            image_path: Path to original image
            mask_path: Path to mask image
            config: Configuration object
            strength: Controls similarity between generated and original images

        Returns:
            Generated pseudo-healthy medical image
        """
        # Load original image and mask
        init_image = Image.open(image_path)
        mask_image = Image.open(mask_path).convert("L")

        # Ensure image has three channels
        init_image = self._ensure_three_channels(init_image)

        # Generate control image
        control_image = self.prepare_control_image2(init_image, mask_image)

        # Generation parameters
        generator = torch.manual_seed(40)

        # Generate image
        result = self.pipe(
            prompt=config.get_prompt(),
            negative_prompt=config.get_negative_prompt(),
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            controlnet_conditioning_scale=1.0,
            strength=config.strength,
            generator=generator,
            num_inference_steps=40,
            guidance_scale=7.5,
            width=256,
            height=256,
            quiet=True  # Add this line to disable progress display
        ).images[0]
        # # For ultrasound images, convert to grayscale to match original image color distribution
        # if config.modality == "ultrasound":
        #     # Convert generated image to grayscale
        #     result = result.convert('L')
        #     # Convert back to RGB to maintain consistent format
        #     result = result.convert('RGB')

        return result

    def visualize_difference(self, orig_img: Image.Image, gen_img: Image.Image, mask: Image.Image,
                             save_path: str, is_show_diff: bool = True) -> None:
        """Visualize original image, generated image, and differences"""
        if is_show_diff:
            # Convert images to numpy
            orig = np.array(orig_img)
            gen = np.array(gen_img)
            mask = np.array(mask).astype(np.uint8)

            # Ensure original image has three channels
            orig = self._ensure_three_channels(Image.fromarray(orig))
            orig = np.array(orig)

            # Calculate difference
            diff = cv2.absdiff(orig, gen)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, diff_binary = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)  # Binarize difference image

            # Create anomaly representation image
            anomaly_overlay = orig.copy()
            anomaly_overlay[diff_binary > 0] = [255, 0, 0]  # Cover difference areas with red

            # Visualize four subplots
            plt.figure(figsize=(20, 5))

            # Original image
            plt.subplot(141)
            plt.imshow(orig)
            plt.axis('off')
            plt.title("Original Image")

            # Generated image
            plt.subplot(142)
            plt.imshow(gen)
            plt.axis('off')
            plt.title("Generated Image")

            # Ground truth mask
            plt.subplot(143)
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.title("GT Mask")

            # Anomaly representation
            plt.subplot(144)
            plt.imshow(anomaly_overlay)
            plt.axis('off')
            plt.title("Anomaly Representation")

            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

    def save_anomaly_visualization(self, orig_img: Image.Image, gen_img: Image.Image,
                                   save_path_overlay: str, save_path_abnormal: str) -> None:
        """
        Save anomaly_overlay image and visualize abnormal_image

        Args:
            orig_img: Original image
            gen_img: Generated image
            save_path_overlay: Path to save anomaly_overlay image
            save_path_abnormal: Path to save abnormal_image
        """
        # Convert images to numpy
        orig = np.array(orig_img)
        gen = np.array(gen_img)

        # Ensure original image has three channels
        orig = self._ensure_three_channels(Image.fromarray(orig))
        orig = np.array(orig)

        # Calculate difference
        diff = cv2.absdiff(orig, gen)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, diff_binary = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        # Create anomaly representation image
        anomaly_overlay = orig.copy()
        anomaly_overlay[diff_binary > 0] = [255, 0, 0]  # Cover difference areas with red

        # Save anomaly_overlay image
        overlay_img = Image.fromarray(anomaly_overlay)
        overlay_img.save(save_path_overlay)

        # Calculate abnormal_image = torch.abs(original_image - generated_image)
        orig_tensor = torch.from_numpy(orig.astype(np.float32)).permute(2, 0, 1)
        gen_tensor = torch.from_numpy(gen.astype(np.float32)).permute(2, 0, 1)
        abnormal_tensor = torch.abs(orig_tensor - gen_tensor)

        # Normalize to 0-255 range
        abnormal_normalized = (abnormal_tensor - abnormal_tensor.min()) / (
                abnormal_tensor.max() - abnormal_tensor.min()) * 255
        abnormal_numpy = abnormal_normalized.permute(1, 2, 0).numpy().astype(np.uint8)

        # Save abnormal_image
        abnormal_img = Image.fromarray(abnormal_numpy)
        abnormal_img.save(save_path_abnormal)


def process_single_image(image_path: str, mask_path: str, output_dir: str, config: Config) -> None:
    """Process a single image file"""
    # Initialize generator
    generator = PseudoHealthGenerator(config)

    # Create output directories
    os.makedirs(os.path.join(output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparison"), exist_ok=True)

    # Get filename (without extension)
    file_name = os.path.splitext(os.path.basename(image_path))[0]

    try:
        # Generate image
        gen_img = generator.generate(image_path, mask_path, config, strength=config.strength)

        # Save generated result
        generated_path = os.path.join(output_dir, "generated", f"{file_name}.png")
        gen_img.save(generated_path)

        # Load original image and mask
        orig_img = Image.open(image_path)
        mask = Image.open(mask_path).convert("L")

        # Visualize differences
        comparison_path = os.path.join(output_dir, "comparison", f"{file_name}.png")
        generator.visualize_difference(
            orig_img, gen_img, mask, comparison_path, is_show_diff=config.is_show_diff
        )

        print(f"Successfully processed file: {file_name}")
        print(f"Generated image saved to: {generated_path}")
        print(f"Comparison image saved to: {comparison_path}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


def process_dataset(config: Config) -> None:
    """Process all PNG image files in the dataset"""
    import glob

    # Initialize generator
    generator = PseudoHealthGenerator(config)

    # Get all PNG image files
    img_dir = os.path.join(config.img_root, "img")
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))

    if not img_paths:
        print(f"No PNG image files found in directory {img_dir}")
        return

    # Create output directories
    os.makedirs(os.path.join(config.output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "comparison"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "diff"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "diff_enhance"), exist_ok=True)

    # Process each image file
    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            # Get filename (without extension)
            file_name = os.path.splitext(os.path.basename(img_path))[0]

            # Construct corresponding mask path
            mask_path = os.path.join(config.mask_root, f"{file_name}.png")

            # Check if mask file exists
            if not os.path.exists(mask_path):
                print(f"Warning: Cannot find corresponding mask file {mask_path}, skipping processing {file_name}")
                continue

            # Generate image
            gen_img = generator.generate(img_path, mask_path, config, strength=config.strength)

            # Save generated result
            gen_img.save(os.path.join(config.output_dir, "generated", f"{file_name}.png"))

            # Load original image and mask
            orig_img = Image.open(img_path)
            mask = Image.open(mask_path).convert("L")

            # Visualize differences
            if config.is_show_diff:
                generator.save_anomaly_visualization(
                    orig_img, gen_img,
                    os.path.join(config.output_dir, "diff_enhance", f"{file_name}.png"),
                    os.path.join(config.output_dir, "diff", f"{file_name}.png")
                )

        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")

if __name__ == "__main__":
    # Example 1: Process a single image
    # config = Config()
    # config.modality = "ct"
    # config.part = "abdomen"
    # config.target_tissue = "liver"
    #
    # # Set to use second prompt template
    # config.set_prompt_template_index(1)
    # config.set_negative_prompt_template_index(1)
    #
    # # Process single image
    # image_path = "../../dataset/RESC_1000_256/img/sample001.png"
    # mask_path = "../results/deeplabv3plus_resnet50_resnet50_RESC_1000_256/coarse_mask/sample001.png"
    # process_single_image(image_path, mask_path, config.output_dir, config)

    # Example 2: Process entire dataset
    config = Config()
    process_dataset(config)
