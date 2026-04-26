# app.py
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import requests
import os
import io

# Define the Generator architecture (same as your training code)
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_act=True, dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2) if use_act else nn.Identity()
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU() if use_act else nn.Identity()
            )
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv(x))

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.down1 = UNetBlock(3, 64, down=True, use_act=False)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        self.down7 = UNetBlock(512, 512)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = UNetBlock(512, 512, down=False, dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        bottleneck = self.bottleneck(d7)
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        
        return self.final(torch.cat([up7, d1], 1))

# Download model from GitHub release
@st.cache_resource
def load_model():
    # URL for your model file (you need to get the direct download link)
    # For GitHub releases, the direct download URL format is:
    # https://github.com/Abdulbaset1/Doodle-to-Real-Image-Translation-and-Colorization-using-Pix2Pix/releases/download/v1/gen_25.pth
    
    model_url = "https://github.com/Abdulbaset1/Doodle-to-Real-Image-Translation-and-Colorization-using-Pix2Pix/releases/download/v1/gen_25.pth"
    model_path = "gen_25.pth"
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Please wait!"):
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            else:
                st.error(f"Failed to download model. Status code: {response.status_code}")
                return None
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        st.success(f"Model loaded successfully on {device}!")
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Preprocess image
def preprocess_image(image, target_size=256):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Transform and add batch dimension
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# Postprocess output
def postprocess_output(tensor):
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu()
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    # Convert to PIL Image
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Sketch to Real Image Generator", layout="wide")
    
    st.title("🎨 Sketch to Real Image Translation")
    st.markdown("Upload a sketch and watch it transform into a realistic image using Pix2Pix GAN!")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Create two columns for input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Sketch")
        uploaded_file = st.file_uploader("Choose a sketch image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display input image
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Sketch", use_container_width=True)
            
            # Add generate button
            if st.button("🚀 Generate Real Image", type="primary"):
                with st.spinner("Generating image... This may take a few seconds."):
                    # Preprocess
                    input_tensor = preprocess_image(input_image)
                    input_tensor = input_tensor.to(device)
                    
                    # Generate
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    # Postprocess
                    output_image = postprocess_output(output_tensor)
                    
                    # Display in second column
                    with col2:
                        st.subheader("✨ Generated Real Image")
                        st.image(output_image, caption="Generated Output", use_container_width=True)
                        
                        # Add download button
                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="💾 Download Generated Image",
                            data=byte_im,
                            file_name="generated_image.png",
                            mime="image/png"
                        )
    
    # Add information in sidebar
    with st.sidebar:
        st.markdown("## 📌 About")
        st.markdown("""
        This app uses a **Pix2Pix GAN** model trained on the CUHK Face Sketch Database (CUFS).
        
        ### How it works:
        1. Upload a sketch/face drawing
        2. The model processes the sketch
        3. Generates a realistic face image
        
        ### Model Architecture:
        - **Generator**: U-Net with skip connections
        - **Training**: Conditional GAN with L1 loss
        - **Input Size**: 256x256 pixels
        - **Output**: RGB image in range [0,1]
        
        ### Note:
        For best results, use front-facing face sketches.
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ using Streamlit & PyTorch")

if __name__ == "__main__":
    main()
