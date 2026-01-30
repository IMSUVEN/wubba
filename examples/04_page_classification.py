#!/usr/bin/env python3
"""Page Classification: Use embeddings for downstream classification.

Use case: Classify pages into categories (e-commerce, blog, news, etc.)
using Wubba embeddings as features for a classifier.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from wubba import Config, WubbaInference


def train_classifier(
    train_html: list[str],
    train_labels: list[str],
    model: WubbaInference,
    dim: int = 128,
) -> LogisticRegression:
    """Trains a classifier on top of Wubba embeddings."""
    embeddings = model.predict(train_html, dim=dim).numpy()

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(embeddings, train_labels)

    return classifier


def evaluate_classifier(
    classifier: LogisticRegression,
    test_html: list[str],
    test_labels: list[str],
    model: WubbaInference,
    dim: int = 128,
) -> dict:
    """Evaluates classifier performance."""
    embeddings = model.predict(test_html, dim=dim).numpy()
    predictions = classifier.predict(embeddings)

    report = classification_report(test_labels, predictions, output_dict=True)
    return report


if __name__ == "__main__":
    # Sample dataset: page type classification
    pages_and_labels = [
        # E-commerce product pages
        (
            "<body><nav><a>Shop</a></nav><main><div class='product'><img/><h1>Laptop</h1><span class='price'>$999</span><button>Add to Cart</button></div></main></body>",
            "product",
        ),
        (
            "<body><nav><a>Shop</a></nav><main><div class='product'><img/><h1>Phone</h1><span class='price'>$599</span><button>Buy Now</button></div></main></body>",
            "product",
        ),
        (
            "<body><header><a>Store</a></header><section class='item'><img/><h2>Headphones</h2><p>$199</p><button>Add</button></section></body>",
            "product",
        ),
        # Blog/article pages
        (
            "<body><header><h1>Blog</h1></header><article><h2>How to Code</h2><p>Content here...</p><p>More text...</p></article><footer>Author</footer></body>",
            "article",
        ),
        (
            "<body><nav><a>Home</a></nav><main><article><h1>Tech News</h1><time>2024-01-01</time><p>Article content...</p></article></main></body>",
            "article",
        ),
        (
            "<body><header><nav><a>Blog</a></nav></header><div class='post'><h1>Tutorial</h1><div class='content'><p>Step 1...</p></div></div></body>",
            "article",
        ),
        # Listing pages
        (
            "<body><nav><a>Browse</a></nav><main><ul class='items'><li>Item 1</li><li>Item 2</li><li>Item 3</li><li>Item 4</li></ul></main></body>",
            "listing",
        ),
        (
            "<body><header><h1>Catalog</h1></header><div class='grid'><div>Card 1</div><div>Card 2</div><div>Card 3</div></div></body>",
            "listing",
        ),
        (
            "<body><nav><a>Search</a></nav><section><div class='results'><article>Result 1</article><article>Result 2</article></div></section></body>",
            "listing",
        ),
        # Form pages
        (
            "<body><main><form><label>Email</label><input type='email'/><label>Password</label><input type='password'/><button>Login</button></form></main></body>",
            "form",
        ),
        (
            "<body><header><h1>Contact</h1></header><form><input placeholder='Name'/><textarea></textarea><button>Send</button></form></body>",
            "form",
        ),
        (
            "<body><div class='signup'><h2>Register</h2><form><input/><input/><input/><button>Submit</button></form></div></body>",
            "form",
        ),
    ]

    pages = [p for p, _ in pages_and_labels]
    labels = [label for _, label in pages_and_labels]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        pages, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Train and evaluate
    model = WubbaInference("models/best.ckpt", Config())

    print("=== Training Classifier ===")
    classifier = train_classifier(X_train, y_train, model, dim=128)

    print("\n=== Evaluation Results ===")
    report = evaluate_classifier(classifier, X_test, y_test, model, dim=128)

    for label in ["product", "article", "listing", "form"]:
        if label in report:
            metrics = report[label]
            print(
                f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}"
            )

    print(f"\nOverall Accuracy: {report['accuracy']:.2f}")

    # Predict new page
    print("\n=== Predicting New Page ===")
    new_page = "<body><nav><a>Shop</a></nav><main><div><img/><h1>New Product</h1><span>$299</span><button>Buy</button></div></main></body>"
    embedding = model.predict([new_page], dim=128).numpy()
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]

    print(f"Predicted: {prediction}")
    print(f"Probabilities: {dict(zip(classifier.classes_, probabilities.round(3), strict=True))}")
