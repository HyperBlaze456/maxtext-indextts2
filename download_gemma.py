import kagglehub

# Download latest version
path = kagglehub.model_download("google/gemma-3/flax/gemma3-4b")

print("Path to model files:", path)