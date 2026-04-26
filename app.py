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
import sys

# Must be first Streamlit command
st.set_page_config(
    page_title="Sketch to Real Image Translator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 Sketch to Real Image Translation")
st.markdown("Upload a sketch and watch it transform into a realistic image using Pix2Pix GAN!")

# Sidebar info
with st.sidebar:
    st.markdown("## 📌 About")
    st.markdown("""
    This app uses a **Pix2Pix GAN** model trained on face sketches.
    
    ### How it works:
    1. Upload a face sketch
    2. AI processes the sketch
    3. Generates realistic face
    
    ### Tips:
    - Use front-facing sketches
    - Clear lines work best
    - Image resized to 256x256
    """)
    st.markdown("---")
    st.markdown(f"🐍 Python: {sys.version.split()[0]}")
    st.markdown(f"🔥 PyTorch: {torch.__version__}")

# Model architecture (same as your training)
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

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    # CORRECTED URL - using download instead of tag
    model_url = "https://github.com/Abdulbaset1/Doodle-to-Real-Image-Translation-and-Colorization-using-Pix2Pix/releases/download/v1/gen_25.pth"
    model_path = "gen_25.pth"
    
    # Download model if not exists
    if not os.path.exists(model_path):
        try:
            with st.spinner("📥 Downloading model (this may take a minute)..."):
                response = requests.get(model_url, timeout=60)
                response.raise_for_status()
                
                with open(model_path, "wb") as f:
                    f.write(response.content)
                
                st.success("✅ Model downloaded!")
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            st.info("Please make sure the model file exists at: gen_25.pth in the release")
            return None, None
    
    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Generator()
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle potential 'module.' prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        st.success(f"✅ Model loaded on {device.upper()}!")
        return model, device
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        return None, None

# Image preprocessing
def preprocess(image, target_size=256):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(image).unsqueeze(0)

# Image postprocessing
def postprocess(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

# Main app
def main():
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Cannot proceed without model. Please check the error above.")
        return
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Sketch")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, use_container_width=True)
            
            if st.button("🚀 Generate", type="primary", use_container_width=True):
                with st.spinner("🎨 Generating..."):
                    # Process
                    input_tensor = preprocess(input_image).to(device)
                    
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    output_image = postprocess(output_tensor)
                    
                    # Display result
                    with col2:
                        st.subheader("✨ Result")
                        st.image(output_image, use_container_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        st.download_button(
                            label="💾 Download",
                            data=buf.getvalue(),
                            file_name="generated.png",
                            mime="image/png",
                            use_container_width=True
                        )

if __name__ == "__main__":
    main()
