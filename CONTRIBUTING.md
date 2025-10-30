# Contributing to DisgraPhi

Thank you for your interest in contributing to DisgraPhi! This project aims to make handwriting recognition accessible to everyone, especially those with dysgraphia and other handwriting challenges.

## Ways to Contribute

### 🐛 Report Bugs
- Use [GitHub Issues](https://github.com/velocitatem/disgraPhi/issues)
- Include: OS, Python version, error messages, steps to reproduce
- Screenshots are helpful!

### 💡 Suggest Features
- Open a [Discussion](https://github.com/velocitatem/disgraPhi/discussions)
- Explain the use case and why it matters
- Consider accessibility implications

### 📝 Improve Documentation
- Fix typos, clarify instructions
- Add examples or tutorials
- Improve the Colab notebook
- Translate documentation

### 🔧 Submit Code
- Fix bugs
- Implement features
- Improve model accuracy
- Optimize performance

### 🧪 Test and Provide Feedback
- Try the Colab notebook with different handwriting styles
- Report what works and what doesn't
- Share accuracy metrics (anonymously)

## Getting Started

### Setup Development Environment

```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/disgraPhi.git
cd disgraPhi

# Create virtual environment
make venv
source .venv/bin/activate

# Install in development mode with dev tools
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Project Structure

```
disgraPhi/
├── ml/                          # Machine learning code
│   ├── models/                  # Model architectures & training
│   │   ├── train.py            # Main training script
│   │   ├── providers/          # Model implementations
│   │   └── eval.py             # Evaluation metrics
│   ├── data/                    # Data processing
│   │   ├── datasets.py         # PyTorch datasets
│   │   ├── etl.py              # Data pipeline
│   │   └── augmentation.py     # Data augmentation
│   └── inference.py            # FastAPI inference server
├── alveslib/                    # Shared utilities
│   ├── logger.py               # Logging utilities
│   └── post_processing.py      # Text post-processing
├── apps/                        # Web applications
│   ├── webapp/                 # Next.js frontend
│   └── webapp-minimal/         # Streamlit app
├── DisgraPhi_Personalization.ipynb  # Colab notebook
└── docs/                        # Documentation
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-123
```

### 2. Make Changes

Follow the code style:
- **Python**: Black formatter (line length 100)
- **Type hints**: Use when possible
- **Docstrings**: Google style for functions/classes
- **Comments**: Explain why, not what

### 3. Test Your Changes

```bash
# Run linters
black ml/ alveslib/ --check
flake8 ml/ alveslib/

# Run tests (when available)
pytest tests/

# Test package building
python -m build --sdist
```

### 4. Commit

Write clear commit messages:
```
Add feature: brief description

Longer explanation of what changed and why.
Fixes #123
```

### 5. Submit Pull Request

1. Push to your fork
2. Open PR against `main` branch
3. Fill out the PR template
4. Wait for review

## Code Guidelines

### Python Style

```python
# Good
def transcribe_image(image_path: str) -> str:
    """Transcribe handwriting from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Transcribed text
    """
    img = Image.open(image_path).convert('RGB')
    return model.generate(pixel_values=img, prompt="Transcribe...", max_new_tokens=128)

# Avoid
def transcribe(p):
    # transcribe image
    img = Image.open(p)
    img = img.convert('RGB')
    text = model.generate(
        pixel_values=img,
        prompt="Transcribe this handwritten text.",
        max_new_tokens=128
    )
    return text
```

### Model Code

Follow the CLAUDE.md tenets:
- Dense, efficient code
- List comprehensions over loops
- Trust the reader
- No unnecessary comments
- One pass over data when possible

### Notebook Code

Keep it accessible:
- Clear explanations
- Progress indicators
- Error handling with helpful messages
- Mobile-friendly displays

## Testing

### Manual Testing Checklist

For Colab notebook changes:
- [ ] Runs on free T4 GPU
- [ ] Clear instructions for each step
- [ ] Error messages are helpful
- [ ] Works with 10, 30, and 100 samples
- [ ] Download produces valid model

For model changes:
- [ ] Doesn't break existing models
- [ ] Accuracy maintained or improved
- [ ] Memory usage reasonable
- [ ] Training time acceptable

For data pipeline:
- [ ] Handles edge cases (missing files, etc.)
- [ ] Proper error messages
- [ ] Caching works correctly
- [ ] Works with different image formats

## Accessibility Considerations

DisgraPhi serves people with disabilities. Keep this in mind:

- **Clear instructions**: Don't assume technical knowledge
- **Error messages**: Explain what went wrong and how to fix it
- **Progress indicators**: Show what's happening during long operations
- **Fallbacks**: Gracefully handle failures
- **Documentation**: Use plain language, avoid jargon

## Model Contributions

### Adding New Models

1. Create provider in `ml/models/providers/`
2. Implement `BaseVisionLanguageModel` interface
3. Update `create_model()` factory
4. Test with training pipeline
5. Document in README

### Improving Accuracy

- Share techniques in Discussions first
- Benchmark against existing models
- Document parameters and trade-offs
- Consider memory/speed impact

## Data Contributions

### Datasets

We cannot accept handwriting datasets due to privacy concerns.

You can contribute:
- Data augmentation techniques
- Data processing improvements
- Support for new data formats

### Packet Templates

Improvements to practice packet generation:
- Better pangrams/text samples
- Multiple languages
- Different difficulty levels
- Accessibility improvements

## Documentation

### What to Document

- New features and APIs
- Configuration options
- Common pitfalls
- Performance tips
- Accessibility features

### Documentation Style

- Start with the problem
- Show examples first
- Explain why, not just how
- Link to related docs
- Keep it up to date

## Review Process

### What We Look For

- ✅ Solves the stated problem
- ✅ Doesn't break existing functionality
- ✅ Includes tests (when applicable)
- ✅ Follows code style
- ✅ Has clear documentation
- ✅ Considers accessibility

### Timeline

- Most PRs reviewed within 3-5 days
- Complex changes may take longer
- We'll provide feedback and work with you

## Questions?

- **General**: [Discussions](https://github.com/velocitatem/disgraPhi/discussions)
- **Bugs**: [Issues](https://github.com/velocitatem/disgraPhi/issues)
- **Security**: Email (see SECURITY.md)

## Code of Conduct

Be respectful and inclusive:
- Welcome newcomers
- Be patient with questions
- Give constructive feedback
- Focus on the code, not the person
- Respect different perspectives

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make handwriting recognition accessible to everyone! 🎉
