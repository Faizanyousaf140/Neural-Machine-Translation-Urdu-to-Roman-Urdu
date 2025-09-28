# 🚀 Deployment Instructions for Urdu-Roman Translator

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Git** for cloning the repository
3. **Trained model files** (best_model.pt, vocab files)

## 🛠️ Local Deployment

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Model Files
Ensure you have these files in your project directory:
- `best_model.pt` (trained model)
- `processed_data/src_vocab.pkl`
- `processed_data/tgt_vocab.pkl`
- `processed_data/src_idx2char.pkl`
- `processed_data/tgt_idx2char.pkl`

### Step 3: Run the Application
```bash
streamlit run app.py
```

The app will be available at: `http://localhost:8501`

## 🌐 Cloud Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/urdu-roman-translator.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy with these settings:
     - **Main file path**: `app.py`
     - **Python version**: 3.8
     - **Requirements file**: `requirements.txt`

### Option 2: Heroku

1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**:
   ```
   python-3.8.16
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Option 3: Docker

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.8-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t urdu-translator .
   docker run -p 8501:8501 urdu-translator
   ```

## 🔧 Configuration

### Environment Variables
Create `.env` file for configuration:
```
MODEL_PATH=best_model.pt
VOCAB_PATH=processed_data/
MAX_TRANSLATION_LENGTH=100
CONFIDENCE_THRESHOLD=0.6
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 📊 Performance Optimization

### For Production:
1. **Model Optimization**:
   - Use `torch.jit.script()` for faster inference
   - Implement model quantization
   - Use ONNX for cross-platform deployment

2. **Caching**:
   - Streamlit's `@st.cache_resource` is already implemented
   - Consider Redis for distributed caching

3. **Monitoring**:
   - Add logging for translation requests
   - Monitor model performance metrics
   - Set up error tracking

## 🚨 Troubleshooting

### Common Issues:

1. **Model Loading Error**:
   - Check file paths
   - Ensure model architecture matches
   - Verify PyTorch version compatibility

2. **Memory Issues**:
   - Reduce batch size
   - Use CPU instead of GPU
   - Implement model pruning

3. **Slow Performance**:
   - Enable model caching
   - Optimize translation function
   - Use smaller model variants

## 📱 Mobile Responsiveness

The app is designed to be mobile-friendly with:
- Responsive layout
- Touch-friendly buttons
- Optimized text input
- Mobile-specific CSS

## 🔒 Security Considerations

1. **Input Validation**:
   - Sanitize user inputs
   - Limit input length
   - Validate character sets

2. **Rate Limiting**:
   - Implement request throttling
   - Add usage quotas
   - Monitor abuse patterns

## 📈 Analytics and Monitoring

The app includes built-in analytics:
- Translation history tracking
- Performance metrics
- User interaction patterns
- Error logging

## 🎯 Production Checklist

- [ ] Model files uploaded
- [ ] Dependencies installed
- [ ] Environment variables set
- [ ] Domain configured (if custom)
- [ ] SSL certificate installed
- [ ] Monitoring set up
- [ ] Backup strategy implemented
- [ ] Performance testing completed

## 📞 Support

For deployment issues:
1. Check the logs
2. Verify file permissions
3. Test locally first
4. Check Streamlit documentation
5. Review error messages

## 🎉 Success!

Once deployed, your Urdu-Roman translator will be live and accessible to users worldwide!
