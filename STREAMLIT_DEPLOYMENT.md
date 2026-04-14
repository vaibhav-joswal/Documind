# Deployment to Streamlit Community Cloud

Your project is now fully configured to run on streamit's Linux environment! Follow these simple steps.

1. **Commit and Push to GitHub:**
   Make sure you commit all files in this directory (including `requirements.txt` and `packages.txt`) and push them to a GitHub repository.

2. **Sign in to Streamlit Share:**
   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3. **Deploy Your App:**
   - Click **"Create app"** -> **"Yup, I have an app"**.
   - Select your repository and the branch you pushed to.
   - For **Main file path**, type `app.py`.
   
4. **Set Your API Key (Secrets):**
   - Before clicking Deploy, click **"Advanced settings"**.
   - In the "Secrets" box, securely add your actual HuggingFace API key like this:
     ```toml
     HF_API_KEY = "hf_YourRealKeyGoesHere"
     ```

5. **Click Deploy!**
   Streamlit will automatically detect `packages.txt` and install Tesseract and Poppler required for OCR. It will then read `requirements.txt` and install your Python dependencies. The app will launch shortly after.
