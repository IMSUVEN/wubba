# --- Training
from wubba.config import Config
from wubba.train import train

config = Config()
train(config=config)


# --- Inferences ---
from wubba.config import Config
from wubba.inference import WubbaInference

model = WubbaInference("models/best.ckpt", Config())

html_docs = [
    "<body><h1>Hello</h1><p>This is a test.</p></body>",
    "<body><nav><a>Home</a></nav></body>",
]

embeddings = model.predict(html_docs)
