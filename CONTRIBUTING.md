# Contributing Guide 🤝

Thank you for considering contributing to Stock Predictor! Your contributions make this project better.

## Code of Conduct

Be respectful and constructive. Treat all contributors with kindness.

## Ways to Contribute

### 1. Report Bugs
- Check if bug already exists
- Create clear, detailed issue
- Include steps to reproduce
- Share error messages and logs

### 2. Suggest Features
- Check if feature is already requested
- Explain use case and benefits
- Provide implementation ideas if possible

### 3. Improve Documentation
- Fix typos and unclear sections
- Add examples and clarifications
- Improve README and guides

### 4. Write Code
- Implement bug fixes
- Add new features
- Improve performance
- Write tests

## Getting Started

### Fork and Clone
```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR-USERNAME/stock-predictor.git
cd stock-predictor
```

### Install Development Dependencies
```bash
pip install -r requirements.txt
pip install pytest pylint black
```

### Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### 1. Make Changes
- Write clean, readable code
- Follow PEP 8 style guide
- Add comments for complex logic
- Keep functions small and focused

### 2. Test Locally
```bash
streamlit run app.py
```

Test your changes in the browser:
- Try different stocks
- Test edge cases
- Check error handling

### 3. Format Code
```bash
# Format with black
black app.py

# Check with pylint
pylint app.py
```

### 4. Commit Changes
```bash
git add .
git commit -m "Clear description of changes"
```

Use conventional commits:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `docs: Update documentation`
- `refactor: Improve code quality`
- `test: Add tests for X`

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Go to GitHub, create Pull Request with:
- Clear title
- Detailed description
- Reference to any related issues
- Screenshots if applicable

## Pull Request Process

1. **Update README.md** with new features (if applicable)
2. **Test thoroughly** - test different scenarios
3. **Keep PR focused** - one feature per PR
4. **Write clear messages** - reviewers will be more helpful
5. **Be responsive** - address feedback promptly
6. **Be patient** - reviews take time

## Code Style

### Python Style Guide (PEP 8)
```python
# Good
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    gains = []
    losses = []
    return rsi_values

# Avoid
def calcRSI(prices,period=14):
 # comments
    g=[]
    l=[]
    return
```

### Naming Conventions
- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Meaningful names (not `a`, `x`, etc.)

### Documentation
```python
def predict_prices(df, days):
    """
    Predict future prices using Linear Regression.
    
    Args:
        df: DataFrame with historical prices
        days: Number of days to predict (7-90)
    
    Returns:
        Tuple of (forecast_df, model)
    """
```

## Feature Ideas

Have ideas for improvements? Here are some areas:

### Analytics
- [ ] Portfolio tracker
- [ ] Compare multiple stocks
- [ ] Backtesting tool
- [ ] Risk analysis

### Predictions
- [ ] LSTM neural networks
- [ ] Prophet time series
- [ ] ARIMA models
- [ ] Ensemble methods

### Features
- [ ] Email price alerts
- [ ] News sentiment analysis
- [ ] Options calculator
- [ ] Trading strategies

### UI/UX
- [ ] Dark mode
- [ ] Customizable indicators
- [ ] Saved watchlist
- [ ] Export reports

## Testing

### Manual Testing Checklist
- [ ] App runs without errors
- [ ] All features work as expected
- [ ] Valid stocks load data
- [ ] Invalid stocks show error
- [ ] Charts render correctly
- [ ] Data export works
- [ ] Mobile friendly

### Automated Testing (Future)
```python
# Tests go in tests/ folder
def test_calculate_rsi():
    data = pd.DataFrame({'Close': prices})
    result = calculate_rsi(data)
    assert len(result) > 0
```

## Common Issues

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt
```

### Git Conflicts
```bash
# Update from main
git fetch origin
git rebase origin/main

# Resolve conflicts, then:
git add .
git rebase --continue
```

### Tests Failing
- Check Python version (3.8+)
- Verify all dependencies installed
- Clear cache: `pip cache purge`
- Reinstall: `pip install -r requirements.txt --force-reinstall`

## Questions?

- 📖 Check [README.md](README.md)
- 🚀 Check [DEPLOYMENT.md](DEPLOYMENT.md)
- 🐛 Open an issue on GitHub
- 💬 Discussion section on GitHub

## Final Notes

- Start with small contributions
- Focus on quality over quantity
- Communicate clearly
- Be open to feedback
- Have fun! 🎉

Thank you for contributing to Stock Predictor! 💚
